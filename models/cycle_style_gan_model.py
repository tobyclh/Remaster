import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import cycle_gan_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torchvision.transforms as transforms
import torchvision.models as models

import copy
from . import stylized


class CycleStyleGANModel(cycle_gan_model.CycleGANModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names += ['G_A_style', 'G_B_style'] # add this for visualization
        self.style_loss_net = stylized.StylizedLoss(opt)


    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = super(CycleStyleGANModel, CycleStyleGANModel).modify_commandline_options(parser, is_train=is_train) # this syntax is weird
        if is_train:
            parser.add_argument('--lambda_style_A', type=float, default=1e-3, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_style_B', type=float, default=1e-3, help='weight for cycle loss (B -> A -> B)')
        return parser

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_style_A = self.opt.lambda_style_A
        lambda_style_B = self.opt.lambda_style_B
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
        # Style loss
        self.loss_G_A_style = self.style_loss_net(self.fake_A, self.real_A, self.real_B) * lambda_style_A
        self.loss_G_B_style = self.style_loss_net(self.fake_B, self.real_B, self.real_A) * lambda_style_B

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_G_A_style + self.loss_G_B_style
        self.loss_G.backward()

