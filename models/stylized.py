
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.view(-1, 1, 1).clone()
        self.std = std.view(-1, 1, 1).clone()

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.view(-1, 1, 1).clone()
        self.std = std.view(-1, 1, 1).clone()

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        return tensor * self.std + self.mean


class ContentLoss(nn.Module):

    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, _input):
        self.output = _input.clone()
        return _input

def gram_matrix(_input):
    a, b, c, d = _input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = _input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, _input):
        self.G = gram_matrix(_input)
        return _input

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    unnormalization = UnNormalize(torch.tensor([0.5, 0.5, 0.5]).cuda(), torch.tensor([0.5, 0.5, 0.5]).cuda())
    normalization = Normalization(normalization_mean, normalization_std).cuda()

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(*[unnormalization, normalization])

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss()
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss()
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


class StylizedLoss(nn.Module):
    def __init__(self, opt, ):
        super().__init__()
        self.opt = opt
        cnn = models.vgg19(pretrained=True).features.cuda().eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).cuda()
        fake_content = fake_style = torch.zeros(1, 3, self.opt.crop_size, self.opt.crop_size).cuda()
        self.model, self.style_losses, self.content_losses = get_style_model_and_losses(cnn, normalization_mean=cnn_normalization_mean, normalization_std=cnn_normalization_std, style_img=fake_style, content_img=fake_content)
        self.weights = torch.Tensor(self.opt.lambda_style_loss).cuda() / sum(self.opt.lambda_style_loss)
        print(f'Style Weights : {self.weights}')

    def forward(self, _input, content, style, weights=None):
        content_features = []
        style_features = []
        input_features = []
        self.model(content)
        for cl in self.content_losses:
            content_features.append(cl.output)
        self.model(style)
        for sl in self.style_losses:
            style_features.append(sl.G)
        if weights is None:
            weights = self.weights
        self.model(_input)
        for cl in self.content_losses:
            input_features.append(cl.output)
        for sl in self.style_losses:
            input_features.append(sl.G)
        score = 0
        for input_feature, target_feature, weight in zip(input_features, content_features + style_features, weights):
            # assert (input_feature - target_feature).abs().mean()> 1e-5, f'input is very close to target, you sure everything right? distance {(input_feature - target_feature).sum()}'
            score += weight * F.mse_loss(input_feature, target_feature)
        return score