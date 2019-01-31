from skimage.io import imread
import torch
from util.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from skimage.color import rgb2lab as sk_rgb2lab
from skimage.color import lab2rgb as sk_lab2rgb
layer = 0
img = imread('botw.jpg')
ori_img = img.copy()
tensor_img = torch.Tensor(img)/255
tensor_lab = sk_rgb2lab(tensor_img.clone())
print(f'tensor_lab : {tensor_lab.shape} {tensor_lab[...,0].min()} {tensor_lab[...,0].max()} {tensor_lab[...,1].min()} {tensor_lab[...,1].max()} {tensor_lab[...,2].min()} {tensor_lab[...,2].max()}')

steps = 6
light_adjusted = [tensor_lab.copy() for i in range(steps)]
for i in range(steps):
    light_adjusted[i][...,layer] *= i/(steps-1)

for i, la in enumerate(light_adjusted):
    plt.subplot(1, steps, i+1)
    print(f'la : {la.shape} {la[...,0].min()} {la[...,0].max()} {la[...,1].min()} {la[...,1].max()} {la[...,2].min()} {la[...,2].max()}')
    out = sk_lab2rgb(la)
    plt.imshow(out)
plt.show()

