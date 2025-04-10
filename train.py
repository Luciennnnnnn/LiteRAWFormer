#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import dotenv

dotenv.load_dotenv(override=True)

import time
import gc
from copy import deepcopy
import argparse
import logging
import math
import os
import random
import shutil
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf
from packaging import version
from tqdm.auto import tqdm

import numpy as np

import torch

import torch.nn.functional as F
import torch.utils.checkpoint

from torchvision.transforms.functional import crop, to_pil_image, to_tensor

import accelerate
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import transformers

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from rawsr.dataset.rawsr_dataset import RAWSRDataset, RAWSRValDataset
from rawsr.utils.logging_utils import TqdmToLogger
from rawsr.utils.raw.utils import raw_to_rgb

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.22.0.dev0")

logger = get_logger(__name__)

@torch.no_grad()
def validation(val_dataset, net, args, accelerator, step, evaluate_ema, process_index, num_processes):
    logger.info("Running validation... ")

    logger.info("Validation loading model... ")
    net = accelerator.unwrap_model(net)
    
    logger.info("Validation images... ")
    index_list = list(range(process_index, len(val_dataset), num_processes))
    
    final_val = step >= args.train.max_train_steps
    result_dir = "val_results" if final_val else f"val_results_{step}"
    if evaluate_ema:
        result_dir = f"{result_dir}_ema"
    raw_dir = os.path.join(args.output_dir, result_dir, 'raw')
    rgb_dir = os.path.join(args.output_dir, result_dir, 'rgb')
    lq_rgb_dir = os.path.join(args.output_dir, result_dir, 'lq_rgb')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(lq_rgb_dir, exist_ok=True)

    for idx in index_list:
        data = val_dataset[idx]
        lq_raw_path, lq_raw, raw_max = data['lq_raw_path'], data['lq_raw'], data['raw_max']

        sr_raw = net(lq_raw.unsqueeze(0).to(accelerator.device))
        sr_raw = torch.clamp(sr_raw * 0.5 + 0.5, 0, 1).squeeze()
        
        sr_rgb = raw_to_rgb(sr_raw)
        sr_rgb = to_pil_image(sr_rgb)

        sr_raw = (sr_raw.permute(1, 2, 0).cpu().numpy() * raw_max).astype(np.uint16)

        lq_rgb = raw_to_rgb(torch.clamp(lq_raw * 0.5 + 0.5, 0, 1))
        lq_rgb = to_pil_image(lq_rgb)

        # save to disk
        name, ext = os.path.splitext(os.path.basename(lq_raw_path))
        np.savez(os.path.join(raw_dir, f"{name}{ext}"), raw=sr_raw, max_val=raw_max)
        sr_rgb.save(os.path.join(rgb_dir, f"{name}_sr.png"))
        lq_rgb.save(os.path.join(lq_rgb_dir, f"{name}.png"))

    logger.info("Finish Validation... ")


def parse_args(root_path):
    parser = argparse.ArgumentParser(description="Simple example of a RAWSR training script.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file.",
    )
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    output_dir = os.path.join(root_path, 'experiments', conf.name)
    
    conf.root_dir = root_path
    conf.output_dir = output_dir
    conf.config_file = args.config
    return conf


def main(args):
    logging_dir = Path(args.output_dir, 'logs')

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        mixed_precision=args.train.mixed_precision,
        log_with=args.logger.log_with,
        project_config=accelerator_project_config,
    )

    num_processes = AcceleratorState().num_processes
    process_index = AcceleratorState().process_index

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.config_file, args.output_dir)
        
        # add file handler
        os.makedirs(logging_dir, exist_ok=True)

        log_file = Path(logging_dir, f'{time.strftime("%Y%m%d-%H%M%S")}.log')

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.logger.addHandler(file_handler)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed, device_specific=args.get('device_specific_seed', False))

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    ema_decay = args.train.get('ema_decay', 0)

    from rawsr.models.mymodel import mymodel

    if args.model.get('pretrained_model_path', None) is not None:
        net = mymodel.from_pretrained(args.model.pretrained_model_path)
    else:
        net = mymodel(
            inp_channels=args.model.net_opt.inp_channels,
            out_channels=args.model.net_opt.out_channels,
            dim=args.model.net_opt.dim,
            num_blocks=args.model.net_opt.num_blocks,
            transposed_attn_heads=args.model.net_opt.transposed_attn_heads,
            ffn_expansion_factor=args.model.net_opt.ffn_expansion_factor,
            bias=args.model.net_opt.bias,
            LayerNorm_type=args.model.net_opt.LayerNorm_type,
        )
    net.train()

    if ema_decay != 0:
        net_ema = deepcopy(net)
        net_ema = EMAModel(net_ema.parameters(), decay=ema_decay, model_cls=type(unwrap_model(net)), model_config=net_ema.config)
        
        net_ema.to(accelerator.device)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.train.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    if args.train.get('benchmark_cudnn', False):
        torch.backends.cudnn.benchmark = True

    if args.train.gradient_checkpointing:
        net.enable_gradient_checkpointing()

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if ema_decay != 0:
                    net_ema.save_pretrained(os.path.join(output_dir, "net_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, 'net'))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if ema_decay != 0:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "net_ema"), type(unwrap_model(net)))
                net_ema.load_state_dict(load_model.state_dict())
                net_ema.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = type(unwrap_model(net)).from_pretrained(input_dir, subfolder="net")

                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    if args.train.scale_lr:
        args.train.learning_rate = (
            args.train.learning_rate * args.train.gradient_accumulation_steps * args.train.batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    logger.info(net)

    total_params = sum(p.numel() for p in net.parameters())
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info(f"Number of parameters (M): {total_params / 1000 / 1000}")
    logger.info(f"Number of trainable parameters (M): {total_trainable_params / 1000 / 1000}")

    # Optimizer creation
    net_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    
    optimizer = optimizer_class(
        net_params,
        lr=args.train.learning_rate,
        betas=(args.train.adam_beta1, args.train.adam_beta2),
        weight_decay=args.train.adam_weight_decay,
        eps=args.train.adam_epsilon,
    )

    logger.info("***** Prepare dataset *****")
    train_dataset = RAWSRDataset(args=args.data)
    val_dataset = RAWSRValDataset(args=args.data)

    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(val_dataset)}")

    if args.seed is not None and args.get('workder_specific_seed', False):
        from rawsr.utils.reproducibility import worker_init_fn
        worker_init_fn = partial(
            worker_init_fn,
            num_processes=AcceleratorState().num_processes,
            num_workers=args.train.dataloader_num_workers,
            process_index=AcceleratorState().process_index,
            seed=args.seed,
            same_seed_per_epoch=args.get("same_seed_per_epoch", False),
        )
    else:
        worker_init_fn = None

    logger.info("***** Prepare dataLoader *****")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train.batch_size,
        num_workers=args.train.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.train.gradient_accumulation_steps)
    if 'max_train_steps' not in args.train:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.train.lr_scheduler == 'timm_cosine':
        from rawsr.optim.scheduler.cosine_lr import CosineLRScheduler

        lr_scheduler = CosineLRScheduler(optimizer=optimizer,
                                         t_initial=args.train.t_initial,
                                         lr_min=args.train.lr_min,
                                         cycle_decay=args.train.cycle_decay,
                                         warmup_t=args.train.warmup_t,
                                         warmup_lr_init=args.train.warmup_lr_init,
                                         warmup_prefix=args.train.warmup_prefix,
                                         t_in_epochs=args.train.t_in_epochs)
    else:
        lr_scheduler = get_scheduler(
            args.train.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.train.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.train.max_train_steps * accelerator.num_processes,
            num_cycles=args.train.lr_num_cycles,
            power=args.train.lr_power,
        )

    logger.info("***** Prepare everything with our accelerator *****")
    net, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        net, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.train.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.train.num_train_epochs = math.ceil(args.train.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = OmegaConf.to_container(args)

        # accelerator.init_trackers("rawsr", config=tracker_config, init_kwargs={"wandb": {"name": args.name}})
        accelerator.init_trackers("rawsr", config=tracker_config)

    # Train!
    total_batch_size = args.train.batch_size * accelerator.num_processes * args.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.train.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.train.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.train.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        file=TqdmToLogger(logger, level=logging.INFO)
    )

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                logger.info(f"***** Wandb log dir: {tracker.run.dir} *****")

    for epoch in range(first_epoch, args.train.num_train_epochs):
        # data_start = time.time()
        if 'max_train_steps' in args.train and global_step >= args.train.max_train_steps:
            break
        for step, batch in enumerate(train_dataloader):
            lq_raw, gt_raw = batch['lq_raw'], batch['gt_raw']
            
            print(f"{lq_raw.dtype=}")
            print(f"{gt_raw.dtype=}")
            with accelerator.accumulate(net):
                model_pred = net(lq_raw)
                
                loss_opt = args.train.get('loss_opt', {})
                loss_func = loss_opt.get('loss_func', 'mse')
                if loss_func == 'mse':
                    loss = F.mse_loss(model_pred.float(), gt_raw.float(), reduction="mean")
                elif loss_func == 'l1':
                    loss = F.l1_loss(model_pred.float(), gt_raw.float(), reduction="mean")
                elif loss_func == 'charbonnier':
                    charbonnier_eps = loss_opt.charbonnier_eps
                    loss = torch.sqrt((model_pred.float() - gt_raw.float()) ** 2 + charbonnier_eps)
                    loss = loss.mean()
                else:
                    raise NotImplementedError(f"Do not support {loss_func}")

                frequency_loss_weight = args.train.get('frequency_loss_weight', 0)
                if frequency_loss_weight > 0:
                    model_pred_freq = torch.fft.fft2(model_pred.float())
                    gt_raw_freq = torch.fft.fft2(gt_raw.float())
                    frequency_loss = abs(model_pred_freq - gt_raw_freq).mean()

                    loss = loss + frequency_loss_weight * frequency_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = list(net_params)
                    accelerator.clip_grad_norm_(params_to_clip, args.train.max_grad_norm)

                optimizer.step()
                if 'timm' in args.train.lr_scheduler:
                    lr_scheduler.step(global_step)
                else:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.train.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

                if frequency_loss_weight > 0:
                    logs['frequency_loss'] = frequency_loss.detach().item()

                if ema_decay != 0:
                    net_ema.step(net.parameters())
                    
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.logger.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.logger.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.logger.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.logger.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    
                    if 'visualization_steps' in args.val and (global_step - 1) % args.val.train_visualization_steps == 0:
                        with torch.no_grad():
                            lq_rgb = batch['lq_rgb'][:3]
                            gt_rgb = batch['lq_rgb'][:3]
                            sr_raw = model_pred[:3]
                            sr_raw = torch.clamp(sr_raw * 0.5 + 0.5, 0, 1)
                            sr_rgb = raw_to_rgb(sr_raw)

                            image = torch.cat([lq_rgb, gt_rgb, sr_rgb], dim=0)

                            from torchvision.utils import make_grid
                            for i in range(3):
                                image_grid = make_grid(image[i::3], nrow=image[i::3].shape[0])
                                image_grid = to_pil_image(image_grid)
                                image_grid.save(os.path.join(args.output_dir, f"train_visualization_{global_step}_{i}png"))

                # if accelerator.is_main_process:
                if args.val.validation_steps != -1 and (global_step - 1) % args.val.validation_steps == 0:
                    if ema_decay != 0:
                        # Store the Controlnet parameters temporarily and load the EMA parameters to perform inference.
                        net_ema.store(net.parameters())
                        net_ema.copy_to(net.parameters())

                        validation(
                            val_dataset,
                            net,
                            args,
                            accelerator,
                            global_step,
                            True,
                            process_index,
                            num_processes
                        )
                        
                        # Switch back to the original transformer parameters.
                        net_ema.restore(net.parameters())

                    validation(
                        val_dataset,
                        net,
                        args,
                        accelerator,
                        global_step,
                        False,
                        process_index,
                        num_processes
                    )
                progress_bar.set_postfix(**logs)
                progress_bar.update(1)

                accelerator.log(logs, step=global_step)

            if 'max_train_steps' in args.train and global_step >= args.train.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
        checkpoints = os.listdir(args.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        if len(checkpoints) > 0 and int(checkpoints[-1].split("-")[1]) < global_step:
            if args.logger.checkpoints_total_limit is not None:
                if len(checkpoints) >= args.logger.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - args.logger.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]

                    logger.info(
                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                    )
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)

            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

        net = accelerator.unwrap_model(net)
        net.save_pretrained(os.path.join(args.output_dir, 'net'))

        if ema_decay != 0:
            net_ema.save_pretrained(os.path.join(args.output_dir, 'net_ema'))
            net_ema.copy_to(net.parameters())

    validation(val_dataset, net, args, accelerator, global_step, False, process_index, num_processes)
    if ema_decay != 0:
        # Store the net parameters temporarily and load the EMA parameters to perform inference.
        net_ema.store(net.parameters())
        net_ema.copy_to(net.parameters())
        validation(val_dataset, net, args, accelerator, global_step, True, process_index, num_processes)
        net_ema.restore(net.parameters())
        
    accelerator.end_training()


if __name__ == "__main__":
    root_path = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args(root_path)
    main(args)