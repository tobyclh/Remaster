from skimage.io import imread
import torch
# from pytorch_colors import hsv_to_rgb, lab_to_rgb, rgb_to_hsv, rgb_to_lab
# from models.color_utils import rgb2hsl, hsl2rgb
from util.color import rgb2xyz
import matplotlib.pyplot as plt
func, inv_func = rgb_to_lab, lab_to_rgb
layer = 0
img = imread('botw.jpg')
ori_img = img.copy()
tensor_img = torch.Tensor(img)/255
tensor_hsl = func(tensor_img.permute(2, 0, 1).unsqueeze(0).clone())
steps = 6
light_adjusted = torch.cat([tensor_hsl.clone() for i in range(steps)])
for i in range(steps):
    light_adjusted[i, layer] *= i/(steps-1)+0.5

tensor_rgb = inv_func(light_adjusted).squeeze().permute(0, 2, 3, 1)
for i in range(steps):
    plt.subplot(steps, 1, i+1)
    plt.imshow(tensor_rgb[i])
plt.show()
