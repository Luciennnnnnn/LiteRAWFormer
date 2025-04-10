import dotenv

dotenv.load_dotenv(override=True)

import argparse
import os
from omegaconf import OmegaConf
from tqdm import tqdm

import imageio

import numpy as np

import torch

from accelerate import Accelerator
from accelerate.state import AcceleratorState

from rawsr.utils.raw.utils import raw_to_rgb
from rawsr.dataset.rawsr_dataset import RAWSRValDataset
from rawsr.models.mymodel import mymodel


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for NTIRE 2025 Challenge on RAW Image SR.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default='datasets/RAWSR/test_in',
        help="Input directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='results/RAWSR/test_out',
        help="Output directory."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default='pretrained_models/NTIRE2025_finnal',
        help="Model directory."
    )
    parser.add_argument(
        "--self_ensemble",
        type=str,
        nargs="+",
        default=['self'],
        help="List of self-ensemble methods to use for testing"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='float32',
        help="Data type to use for testing"
    )
    
    args = parser.parse_args()
    return args


def predict(net, lq_raw, method):
    if method == 'self':
        return net(lq_raw)
    elif method == 'horizontal_flip':
        return net(lq_raw.flip(dims=[3])).flip(dims=[3])
    elif method == 'vertical_flip':
        return net(lq_raw.flip(dims=[2])).flip(dims=[2])
    elif method == 'rot90':
        return net(lq_raw.rot90(k=1, dims=[2, 3])).rot90(k=-1, dims=[2, 3])
    elif method == 'rot180':
        return net(lq_raw.rot90(k=2, dims=[2, 3])).rot90(k=-2, dims=[2, 3])
    elif method == 'rot270':
        return net(lq_raw.rot90(k=3, dims=[2, 3])).rot90(k=-3, dims=[2, 3])
    else:
        raise ValueError(f"Invalid method: {method}")

def main(args, root_dir):
    accelerator = Accelerator(mixed_precision=args.dtype if args.dtype != 'fp32' else 'no')

    net = mymodel.from_pretrained(args.model_dir)
    net = net.to(accelerator.device)
    net.eval()

    net = accelerator.prepare(net)
    net = accelerator.unwrap_model(net)

    test_dataset = RAWSRValDataset(
        args=OmegaConf.create(
            {
                "val_data_dir": args.input_dir,
            }
        )
    )
    print(f"{len(test_dataset)=}")
    data_index_list = list(range(AcceleratorState().process_index, len(test_dataset), AcceleratorState().num_processes))

    raw_dir = os.path.join(args.output_dir, 'raw')
    rgb_dir = os.path.join(args.output_dir, 'rgb')

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    with tqdm(total=len(data_index_list), desc=f'process_index {AcceleratorState().process_index}: Processing {len(data_index_list)}/{len(test_dataset)}', unit='image') as pbar:
        for idx in data_index_list:
            data = test_dataset[idx]

            lq_raw = data['lq_raw']
            lq_raw_path = data['lq_raw_path']
            digital_gain = data['digital_gain'] if 'digital_gain' in data else 3.0

            raw_max = data['raw_max']

            name, ext = os.path.splitext(os.path.basename(lq_raw_path))

            lq_raw = lq_raw.to(accelerator.device)
            with torch.inference_mode():
                sr_raws = []
                for method in args.self_ensemble:
                    sr_raw = predict(net, lq_raw.unsqueeze(0), method)
                    sr_raws.append(sr_raw)
                sr_raw = torch.mean(torch.cat(sr_raws, dim=0), dim=0)

            sr_raw = torch.clamp(sr_raw * 0.5 + 0.5, 0, 1).squeeze()
            sr_rgb = raw_to_rgb(sr_raw, digital_gain=digital_gain)
            sr_rgb = sr_rgb * 255
            sr_rgb = sr_rgb.permute(1, 2, 0).cpu().numpy().round().astype(np.uint8)

            sr_raw = (sr_raw.permute(1, 2, 0).cpu().numpy() * raw_max).astype(np.uint16)

            # save to disks
            np.savez(os.path.join(raw_dir, f"{name}{ext}"), raw=sr_raw, max_val=raw_max)
            imageio.imwrite(os.path.join(rgb_dir, f"{name}.png"), sr_rgb)

            pbar.update(1)

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args()
    main(args, root_dir)