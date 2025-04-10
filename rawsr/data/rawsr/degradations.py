import os
import pickle

import numpy as np

from .blur import apply_psf, add_blur
from .noise import add_natural_noise, add_gnoise, add_heteroscedastic_gnoise
from .imutils import downsample_raw, convert_to_tensor

import torch


class RawSRDegradationPipeline(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.kernels = np.load(os.path.join(os.path.dirname(__file__), 'kernels.npy'), allow_pickle=True)

        with open(os.path.join(os.path.dirname( __file__), "psf.pkl"), 'rb') as pklfile:
            self.pyblur_psf_kernels = pickle.load(pklfile, encoding='latin1')


    def forward(self, img):
        """
        Pipeline to add synthetic degradations to a (RAW/RGB) image.
        y = down(x * k) + n
        """

        img = convert_to_tensor(img)
        
        # Apply psf blur: x * k
        img = add_blur(img, self.kernels,
                       pyblur_psf_kernels=self.pyblur_psf_kernels,
                       kernel_list=self.args.kernel_list,
                       random_rotation=self.args.get('random_rotation', False),
                       psf_kernel_prob=self.args.get('psf_kernel_prob', 0.5),
                       pyblur_psf_kernel_prob=self.args.get('pyblur_psf_kernel_prob', 0.0))

        # Apply downsampling down(x*k)
        img = downsample_raw(img)
        
        # Add noise down(x*k) + n
        p_noise = np.random.rand()
        if p_noise > 0.3:
            img = add_natural_noise(img)
        else:
            img = add_heteroscedastic_gnoise(img)
        
        return img


class RawSRDegradationPipelinev2(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.kernels = np.load(os.path.join(os.path.dirname(__file__), 'kernels.npy'), allow_pickle=True)

    def forward(self, img):
        """
        Pipeline to add synthetic degradations to a (RAW/RGB) image.
        y = down(x * k) + n
        """

        img = convert_to_tensor(img)

        # Apply psf blur: x * k
        img = add_blur(img, self.kernels,
                       psf_version=self.args.psf_version,
                       random_rotation=self.args.get('random_rotation', False))

        # Apply downsampling down(x*k)
        img = downsample_raw(img)
        
        # Add noise down(x*k) + n
        p_noise = np.random.rand()
        if p_noise > 0.3:
            img = add_natural_noise(img)
        else:
            img = add_heteroscedastic_gnoise(img)
        
        return img