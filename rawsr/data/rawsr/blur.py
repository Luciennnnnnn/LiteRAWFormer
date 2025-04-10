from typing import Any, Optional, Sequence, TypeVar, cast, List
from typing_extensions import TypeGuard

import math
import random

import numpy as np
from scipy import signal

import torch

from .realesrgan_degradation import bivariate_Gaussian


import kornia

def augment_kernel(kernel, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    Rotate kernels (or images)
    '''
    if mode == 0:
        return kernel
    elif mode == 1:
        return np.flipud(np.rot90(kernel))
    elif mode == 2:
        return np.flipud(kernel)
    elif mode == 3:
        return np.rot90(kernel, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(kernel, k=2))
    elif mode == 5:
        return np.rot90(kernel)
    elif mode == 6:
        return np.rot90(kernel, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(kernel, k=3))
    
    
def apply_custom_filter(inp_img, kernel_):
    return kornia.filters.filter2d(inp_img, kernel_, normalized=True)


def generate_gkernel(ker_sz=None, sigma=None):
    gkern1 = signal.windows.gaussian(ker_sz, std=sigma[0]).reshape(ker_sz, 1)
    gkern2 = signal.windows.gaussian(ker_sz, std=sigma[1]).reshape(ker_sz, 1)
    gkern  = np.outer(gkern1, gkern2)
    return gkern


def apply_gkernel(inp_img, ker_sz=5, ksigma_vals=[.05 + i for i in range(5)],
                  random_rotation: bool = False,
                  rotation_range = (-math.pi, math.pi)):
    """
    Apply uniform gaussian kernel of sizes between 5 and 11.
    """
    # sample for variance
    sigma_val1 = ksigma_vals[np.random.randint(len(ksigma_vals))]
    sigma_val2 = ksigma_vals[np.random.randint(len(ksigma_vals))]
    sigma = (sigma_val1, sigma_val2)
    
    if random_rotation:
        theta = np.random.uniform(rotation_range[0], rotation_range[1])
        kernel = bivariate_Gaussian(ker_sz, sigma[0], sigma[1], theta, isotropic=False)
    else:
        kernel = generate_gkernel(ker_sz, sigma)
    tkernel = torch.from_numpy(kernel.copy()).view(1, ker_sz, ker_sz).type(torch.FloatTensor)
    blurry = apply_custom_filter(inp_img, tkernel).squeeze(0)
    return torch.clamp(blurry, 0., 1.), kernel
    

def apply_psf(inp_img, kernels, kernel_list:List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    """
    Apply PSF
    """
    kernel_p = np.array([0.15, 0.20, 0.20, 0.0075, 0.0075, 0.175, 0.175, 0.05, 0.0075, 0.0075, 0.02])
    kernel_p = kernel_p[kernel_list]
    kernel_p = kernel_p / np.sum(kernel_p)

    idx = np.random.choice(kernel_list, p=kernel_p)

    # if psf_version == 'v1':
    #     idx = np.random.choice(
    #         np.arange(11),
    #         p=[0.15, 0.20, 0.20, 0.0075, 0.0075, 0.175, 0.175, 0.05, 0.0075, 0.0075, 0.02],
    #     )
    # elif psf_version == 'v2':
    #     idx = np.random.choice(
    #         np.arange(12),
    #         p=[0.15, 0.20, 0.20, 0.0075, 0.0075, 0.175, 0.175, 0.05, 0.0075, 0.0075, 0.0125, 0.0075],
    #     )

    kernel = kernels[idx].astype(np.float64)
    kernel = augment_kernel(kernel, mode=random.randint(0, 7))
    ker_sz = 25
    tkernel = torch.from_numpy(kernel.copy()).view(1, ker_sz, ker_sz).type(torch.FloatTensor)
    blurry = apply_custom_filter(inp_img, tkernel).squeeze(0)
    return torch.clamp(blurry, 0., 1.), kernel

def apply_pyblur_psf(inp_img, kernels):
    kernel = random.choice(kernels)
    kernel = augment_kernel(kernel, mode=random.randint(0, 7))
    ker_sz = kernel.shape[-1]
    tkernel = torch.from_numpy(kernel.copy()).view(1, ker_sz, ker_sz).type(torch.FloatTensor)
    blurry = apply_custom_filter(inp_img, tkernel).squeeze(0)
    return torch.clamp(blurry, 0., 1.), kernel


def add_blur(inp_img, kernels, pyblur_psf_kernels, gkern_szs= [3, 5, 7, 9],
             kernel_list: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             random_rotation: bool = False,
             psf_kernel_prob: float = 0.5,
             pyblur_psf_kernel_prob: float = 0.0):
    
    gkernel_prob = 1 - psf_kernel_prob - pyblur_psf_kernel_prob

    # 根据概率决定使用哪种kernel
    p = random.random()
    if p < gkernel_prob:
        ker_sz = gkern_szs[np.random.randint(len(gkern_szs))]
        blurry, kernel = apply_gkernel(inp_img.unsqueeze(0), ker_sz=ker_sz,
                                       random_rotation=random_rotation)
    elif p < gkernel_prob + psf_kernel_prob:
        blurry, kernel = apply_psf(inp_img.unsqueeze(0), kernels, kernel_list=kernel_list)
    else:
        blurry, kernel = apply_pyblur_psf(inp_img.unsqueeze(0), pyblur_psf_kernels)
        
    return blurry


if __name__ == "__main__":
    pass