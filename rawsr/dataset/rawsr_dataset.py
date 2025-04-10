from glob import glob

import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from ..utils.raw.utils import raw_to_rgb


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
        lq_raw_image = to_tensor(lq_raw_image)

        lq_rgb_image = raw_to_rgb(lq_raw_image)

        lq_raw_image = lq_raw_image * 2 - 1
        lq_rgb_image = lq_rgb_image * 2 - 1

        return {'lq_raw': lq_raw_image, 'lq_rgb': lq_rgb_image,
                'raw_max': lq_raw_max_val,
                'lq_raw_path': lq_raw_path}

    def __len__(self):
        return len(self.lq_raw_files)