import dotenv

dotenv.load_dotenv(override=True)

import gc
import time

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

        
def start_timer():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

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
    
    data = test_dataset[0]

    lq_raw = data['lq_raw']
    digital_gain = data['digital_gain'] if 'digital_gain' in data else 3.0
    raw_max = data['raw_max']

    lq_raw = lq_raw.to(accelerator.device)

    from fvcore.nn import parameter_count, FlopCountAnalysis
    print(f"params (M): {parameter_count(net)[''] / 1000000}")
    print(f"GFLOPS: {FlopCountAnalysis(net, lq_raw.unsqueeze(0)).total() / 1000000000}")

    # warmup
    for i in range(10):
        with torch.inference_mode():
            sr_raws = []
            for method in args.self_ensemble:
                sr_raw = predict(net, lq_raw.unsqueeze(0), method)
                sr_raws.append(sr_raw)
            sr_raw = torch.mean(torch.cat(sr_raws, dim=0), dim=0)

    start_timer()
    time_consume = []

    for i in range(20):
        start_time = time.time()
        
        with torch.inference_mode():
            sr_raws = []
            for method in args.self_ensemble:
                sr_raw = predict(net, lq_raw.unsqueeze(0), method)
                sr_raws.append(sr_raw)
            sr_raw = torch.mean(torch.cat(sr_raws, dim=0), dim=0)

        torch.cuda.synchronize()
        end_time = time.time()
        time_consume.append(end_time - start_time)

    time_consume = np.array(time_consume)
    time_consume = np.sort(time_consume)  # Remove 2 min and 2 max values
    print(f"{time_consume=}")
    time_consume = np.mean(time_consume[2:-2])  # Take average of remaining values

    print(f"runtime {time_consume} s")

    # sr_raw = torch.clamp(sr_raw * 0.5 + 0.5, 0, 1).squeeze()
    # sr_rgb = raw_to_rgb(sr_raw, digital_gain=digital_gain)
    # sr_rgb = sr_rgb * 255
    # sr_rgb = sr_rgb.permute(1, 2, 0).cpu().numpy().round().astype(np.uint8)

    # sr_raw = (sr_raw.permute(1, 2, 0).cpu().numpy() * raw_max).astype(np.uint16)

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args()
    main(args, root_dir)