from skimage.io import imread
import torch
# from pytorch_colors import hsv_to_rgb, lab_to_rgb, rgb_to_hsv, rgb_to_lab
# from models.color_utils import rgb2hsl, hsl2rgb
from util.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from skimage.color import rgb2lab as sk_rgb2lab
from skimage.color import lab2rgb as sk_lab2rgb
layer = 0
img = imread('botw.jpg')
ori_img = img.copy()
tensor_img = torch.Tensor(img)/255
tensor_lab = sk_rgb2lab(tensor_img.clone())
plt.subplot(2, 2, 1)
plt.imshow(tensor_lab)
plt.subplot(2, 2, 2)
tensor_rgb = sk_lab2rgb(tensor_lab.copy())
plt.imshow(tensor_rgb)
print(f'tensor_lab : {tensor_lab.shape} {tensor_lab[...,0].min()} {tensor_lab[...,0].max()} {tensor_lab[...,1].min()} {tensor_lab[...,1].max()} {tensor_lab[...,2].min()} {tensor_lab[...,2].max()}')
print(f'tensor_rgb : {tensor_rgb.shape} {tensor_rgb[...,0].min()} {tensor_rgb[...,0].max()} {tensor_rgb[...,1].min()} {tensor_rgb[...,1].max()} {tensor_rgb[...,2].min()} {tensor_rgb[...,2].max()}')
tensor_lab = rgb2lab(tensor_img.clone())
plt.subplot(2, 2, 3)
plt.imshow(tensor_lab.clone())
tensor_rgb = lab2rgb(tensor_lab.clone())
print(f'tensor_lab : {tensor_lab.shape} {tensor_lab[...,0].min()} {tensor_lab[...,0].max()} {tensor_lab[...,1].min()} {tensor_lab[...,1].max()} {tensor_lab[...,2].min()} {tensor_lab[...,2].max()}')
print(f'tensor_rgb : {tensor_rgb.shape} {tensor_rgb[...,0].min()} {tensor_rgb[...,0].max()} {tensor_rgb[...,1].min()} {tensor_rgb[...,1].max()} {tensor_rgb[...,2].min()} {tensor_rgb[...,2].max()}')
plt.subplot(2, 2, 4)
plt.imshow(tensor_rgb.clone())
plt.show()
steps = 6
