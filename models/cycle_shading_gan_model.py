import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import cycle_gan_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models

import copy
from . import stylized
from . import color_utils
from . import networks
# from pytorch_colors import rgb_to_lab, lab_to_rgb
from skimage import color  # require skimage

class CycleShadingGANModel(cycle_gan_model.CycleGANModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # self.loss_names += ['G_A_shading', 'G_B_shading'] # add this for visualization
        
        self.visual_names = ['real_A_rgb', 'real_B_rgb', 'fake_A_rgb', 'fake_B_rgb']
        self.netG_A = networks.define_G(opt.input_nc, 1, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, 1, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(3, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(3, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        fake_B = self.netG_A(self.real_A)                 # G_A(A), fake_B is a tone mapper
        self.fake_B = self.real_A.clone()
        self.fake_B[:, 0:1] += fake_B                   # apply the tone, now actual_fake_B should have the content of one zelda game but the lighting of the other
        self.fake_B.clamp_(-1, 1)
        self.rec_A = self.netG_B(self.fake_B)                # G_B(G_A(A))
        fake_A = self.netG_B(self.real_B)
        self.fake_A = self.real_B.clone()
        self.fake_A[:, 0:1] += fake_A
        self.fake_A.clamp_(-1, 1)
        self.rec_B = self.netG_A(self.fake_A)                # G_A(G_B(B))

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # lambda_shading_A = self.opt.lambda_shading_A
        # lambda_shading_B = self.opt.lambda_shading_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
                                
    @torch.no_grad()
    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        self.real_A_rgb = self.lab2rgb(self.real_A[:, 0:1], self.real_A[:, 1:])
        self.fake_A_rgb = self.lab2rgb(self.fake_A[:, 0:1], self.fake_A[:, 1:])
        self.real_B_rgb = self.lab2rgb(self.real_B[:, 0:1], self.real_B[:, 1:])
        self.fake_B_rgb = self.lab2rgb(self.fake_B[:, 0:1], self.fake_B[:, 1:])

    @torch.no_grad()
    def lab2rgb(self, L, AB):
        """Convert an Lab tensor image to a RGB numpy output
        Parameters:
            L  (1-channel tensor array): L channel images (range: [-1, 1], torch tensor array)
            AB (2-channel tensor array):  ab channel images (range: [-1, 1], torch tensor array)

        Returns:
            rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
        """
        AB2 = AB * 110.0
        L2 = (L + 1.0) * 50.0
        # print(f'L2, AB2 : {L2.shape}, {AB2.shape}')
        Lab = torch.cat([L2, AB2], dim=1)
        Lab = Lab[0].data.cpu().float().numpy()
        Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
        rgb = color.lab2rgb(Lab) * 255
        return rgb
