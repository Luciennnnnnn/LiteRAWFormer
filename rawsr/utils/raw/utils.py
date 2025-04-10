import numpy as np

import torch

from torch.nn import functional as F

import kornia


def gamma_compression(image, gamma_coef: float = 1 / 2.2):
    """Converts from linear to gamma space."""
    return torch.clamp_min(image, 1e-8) ** gamma_coef


def tonemap(image):
    """Simple S-curved global tonemap"""
    return (3 * (image**2)) - (2 * (image**3))


def raw_to_rgb(raw_img, digital_gain: int = 3, gamma_coef: float = 1 / 2.2, cfa = kornia.color.CFA.BG):
    raw_img = F.pixel_shuffle(raw_img, 2)
    raw_img = torch.clamp(raw_img * digital_gain, 0, 1)
    
    if raw_img.dim() < 4:
        rgb_img = kornia.color.raw_to_rgb(raw_img.unsqueeze(0), cfa=cfa)
        rgb_img = rgb_img.squeeze(0)
    else:
        rgb_img = kornia.color.raw_to_rgb(raw_img, cfa=cfa)
        
    rgb_img = gamma_compression(rgb_img, gamma_coef)
    rgb_img = tonemap(rgb_img)
    rgb_img = torch.clamp(rgb_img, 0, 1)

    return rgb_img


def bayer_unification(bayer_arrays, args):
    if not isinstance(bayer_arrays, list):
        bayer_arrays = [bayer_arrays]
    
    for i in range(len(bayer_arrays)):
        if args.bayer_pattern_out == 'BGGR':
            if args.bayer_pattern_in == 'GRBG':
                bayer_arrays[i] = bayer_arrays[i][1:-1, :]
            elif args.bayer_pattern_in == 'GBRG':
                bayer_arrays[i] = bayer_arrays[i][:, 1:-1]
            elif args.bayer_pattern_in == 'RGGB':
                bayer_arrays[i] = bayer_arrays[i][1:-1, 1:-1]
        elif args.bayer_pattern_out == 'RGGB':
            if args.bayer_pattern_in == 'GRBG':
                bayer_arrays[i] = bayer_arrays[i][:, 1:-1]
            elif args.bayer_pattern_in == 'GBRG':
                bayer_arrays[i] = bayer_arrays[i][1:-1, :]
            elif args.bayer_pattern_in == 'BGGR':
                bayer_arrays[i] = bayer_arrays[i][1:-1, 1:-1]
        else:
            raise NotImplementedError(f"Do not support training bayer pattern {args.bayer_pattern_out}")

    if len(bayer_arrays) == 1:
        bayer_arrays = bayer_arrays[0]
    return bayer_arrays


def pack_raw(x):
    return np.stack((x[::2, ::2], x[::2, 1::2],
                     x[1::2, ::2], x[1::2, 1::2]), axis=2)


def unpack_raw(x):
    out = np.zeros((x.shape[0] * 2, x.shape[1] * 2))

    out[::2, ::2] = x[..., 0]
    out[::2, 1::2] = x[..., 1]
    out[1::2, ::2] = x[..., 2]
    out[1::2, 1::2] = x[..., 3]

    return out