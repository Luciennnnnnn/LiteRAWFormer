from glob import glob

import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from ..data.rawsr.degradations import RawSRDegradationPipeline

from ..utils.raw.utils import raw_to_rgb, bayer_unification


class RAWSRDataset(Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.gt_raw_files = sorted(glob(f"{args.train_data_dir}/*.npz"))
        self.degradation_pipeline = RawSRDegradationPipeline(args)

    def __getitem__(self, index):
        gt_raw_file = np.load(self.gt_raw_files[index])
        gt_raw_image, gt_raw_max_val = gt_raw_file['raw'].astype(np.float32), gt_raw_file["max_val"]
        gt_raw_image = bayer_unification(gt_raw_image, self.args)
        # ------------------------------- random crop ------------------------------- #
        top = random.randint(0, (gt_raw_image.shape[0] - self.args.gt_size - 1 - 2) // 2) * 2
        left = random.randint(0, (gt_raw_image.shape[1] - self.args.gt_size - 1 - 2) // 2) * 2

        # since random flip results in 2 pixel loss, we crop extra 2 pixel in each dimention
        gt_raw_image = gt_raw_image[top: top + self.args.gt_size + 2, left: left + self.args.gt_size + 2, :]

        # ------------------------------- random flip ------------------------------- #
        # Horizental Flip
        if np.random.uniform() > 0.5:
            gt_raw_image = gt_raw_image[:, ::-1, :][:, 1:-1, :]
        else:
            if np.random.uniform() > 0.5:
                gt_raw_image = gt_raw_image[:, 2:, :]
            else:
                gt_raw_image = gt_raw_image[:, :-2, :]

        # Vertiacal Flip:
        if np.random.uniform() > 0.5:
            gt_raw_image = gt_raw_image[::-1, :, :][1:-1, :, :]
        else:
            if np.random.uniform() > 0.5:
                gt_raw_image = gt_raw_image[2:, :, :]
            else:
                gt_raw_image = gt_raw_image[:-2, :, :]

        # rot90
        if np.random.uniform() > 0.5:
            gt_raw_image = gt_raw_image.transpose(1, 0, 2)

        # normalization
        gt_raw_image = gt_raw_image / gt_raw_max_val
        gt_raw_image = to_tensor(gt_raw_image).to(torch.float32)

        lq_raw_image = self.degradation_pipeline(gt_raw_image)
        lq_rgb_image = raw_to_rgb(lq_raw_image)
        gt_rgb_image = raw_to_rgb(gt_raw_image)

        lq_raw_image = lq_raw_image * 2 - 1
        gt_raw_image = gt_raw_image * 2 - 1

        lq_rgb_image = lq_rgb_image * 2 - 1
        gt_rgb_image = gt_rgb_image * 2 - 1

        return {'lq_raw': lq_raw_image, 'gt_raw': gt_raw_image,
                'lq_rgb': lq_rgb_image, 'gt_rgb': gt_rgb_image,
                'raw_max': gt_raw_max_val}

    def __len__(self):
        return len(self.gt_raw_files)
    

class RAWSRValDataset(Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.lq_raw_files = sorted(glob(f"{args.val_data_dir}/*.npz"))

    def __getitem__(self, index):
        lq_raw_path = self.lq_raw_files[index]
        lq_raw_file = np.load(lq_raw_path)
        lq_raw_image, lq_raw_max_val = lq_raw_file['raw'].astype(np.float32), lq_raw_file["max_val"]

        lq_raw_image = lq_raw_image / lq_raw_max_val
        lq_raw_image = to_tensor(lq_raw_image).to(torch.float32)

        lq_rgb_image = raw_to_rgb(lq_raw_image)

        lq_raw_image = lq_raw_image * 2 - 1
        lq_rgb_image = lq_rgb_image * 2 - 1

        return {'lq_raw': lq_raw_image, 'lq_rgb': lq_rgb_image,
                'raw_max': lq_raw_max_val,
                'lq_raw_path': lq_raw_path}

    def __len__(self):
        return len(self.lq_raw_files)