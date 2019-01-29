import torch
import pytorch_colors


def rgb2hsl(rgb_tensor):
    """"""
    assert rgb_tensor.min() >= 0 and rgb_tensor.max() <= 1
    hsv_tensor = pytorch_colors.rgb_to_hsv(rgb_tensor)
    # print(f'RGB2HSL : {type(hsv_tensor)}')
    H, S, V = hsv_tensor[:, 0], hsv_tensor[:, 1], hsv_tensor[:, 2]
    L = V - V*S/2
    L_map = (1 - ((L.abs() < 1e-5) | ((L-1).abs()< 1e-5)).float())
    S = (V - L)/ torch.min(L, 1-L)
    S.mul_(L_map)
    return torch.stack([H, S, L], dim=1)

def hsl2rgb(hsl_tensor):
    """"""
    # assert hsl_tensor.min() >= 0 and hsl_tensor.max() <= 1
    H, S, L = hsl_tensor[:, 0], hsl_tensor[:, 1], hsl_tensor[:, 2]
    V = L + S * torch.min(L, 1-L)
    L_map = (1 - (V.abs()< 1e-5).float())
    S = 2 - 2*L/V
    S.mul_(L_map)
    hsv_tensor = torch.stack([H, S, V],dim=1)
    return pytorch_colors.hsv_to_rgb(hsv_tensor)

