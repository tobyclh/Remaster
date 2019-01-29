import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from data.video_dataset import VideoDatasets
from PIL import Image
import random
from skimage import color  # require skimage
import numpy as np
import torchvision.transforms as transforms

class UnalignedVideoDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = VideoDatasets(self.dir_A, return_paths=True)   # load images from '/path/to/data/trainA'
        self.B_paths = VideoDatasets(self.dir_B, return_paths=True)   # load images from '/path/to/data/trainB'
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.color_space = 'lab' if opt.model=='cycle_shading_gan' else 'rgb'
        print(f'Using color space : {self.color_space}')
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1), convert=self.color_space=='rgb')
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1), convert=self.color_space=='rgb')

    def __getitem__(self, index):
        # print(f'Index : {index}')
        A_img, A_path = self.A_paths[index[0], index[1]]
        B_img, B_path = self.B_paths[index[2], index[3]]
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        if self.color_space == 'lab':
            lab_A = color.rgb2lab(A).astype(np.float32)
            lab_A_t = transforms.ToTensor()(lab_A)
            lab_A_t[0, ...] /= 50.0
            lab_A_t[0, ...] -= 1
            lab_A_t[[1,2], ...] /= 110.0
            lab_B = color.rgb2lab(B).astype(np.float32)
            lab_B_t = transforms.ToTensor()(lab_B)
            lab_B_t[0, ...] /= 50.0
            lab_B_t[0, ...] -= 1
            lab_B_t[[1,2], ...] /= 110.0
            A = lab_A_t
            B = lab_B_t
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
