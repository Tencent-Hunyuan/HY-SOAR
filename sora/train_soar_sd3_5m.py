#!/usr/bin/env python
# coding=utf-8
"""
HY-SOAR: Self-Correction for Optimal Alignment and Refinement in diffusion models.

Train SD3.5-Medium with SOAR v4 auxiliary supervision.
Images are loaded directly (no pre-cached latents) and encoded through
the VAE on-the-fly.

Usage (from repo root):
    accelerate launch -m sora.train_soar_sd3_5m \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-medium \
        --jsonl_path /path/to/train.jsonl \
        --image_dir /path/to/images \
        --output_dir ./output/soar_sd3_5m \
        --train_batch_size 4 \
        --max_train_steps 5000 \
        --learning_rate 2e-5 \
        --mixed_precision bf16
"""

import argparse
import copy
import logging
import math
import os
import random
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import is_wandb_available

from sora.utils.algorithm import single_step_aux_points, t_to_sigma_timestep
from sora.utils.data import build_bucket_dataloader
from sora.utils.model import (
    encode_prompt,
    import_model_class_from_model_name_or_path,
    make_load_model_hook,
    make_save_model_hook,
    unwrap_model,
)

if is_wandb_available():
    import wandb  # noqa: F401  pylint: disable=unused-import

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="HY-SOAR training for SD3.5-Medium")

    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)

    # Data
    parser.add_argument("--jsonl_path", type=str, required=True,
                        help="JSONL with md5, caption_en, bw, bh")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing images named by md5")
    parser.add_argument("--max_sequence_length", type=int, default=77)
    parser.add_argument("--random_flip", action="store_true")

    # Training
    parser.add_argument("--output_dir", type=str, default="output/soar_sd3_5m")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")

    # Loss / timestep
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)

    # SOAR v4 hyperparameters
    parser.add_argument("--lambda_aux", type=float, default=1.0,
                        help="Weight for auxiliary loss")
    parser.add_argument("--num_rollout_paths", type=int, default=1,
                        help="Number of rollout paths (1=ODE only, >1 adds stochastic)")
    parser.add_argument("--trajectory_length", type=int, default=6,
                        help="Auxiliary points per rollout path")
    parser.add_argument("--sde_rollout_type", type=str, default="flow_sde",
                        choices=["simple", "sde", "flow_sde", "cps"])
    parser.add_argument("--sde_noise_scale", type=float, default=0.5)
    parser.add_argument("--cfg_scale_sampling", type=float, default=4.5,
                        help="CFG scale for rollout velocity")
    parser.add_argument("--num_sampling_steps", type=int, default=40)
    parser.add_argument("--cond_dropout_rate", type=float, default=0.0,
                        help="Probability of dropping condition (for CFG training)")
    parser.add_argument("--sigma_upper_ratio", type=float, default=1.5)

    # Checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Logging
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default=None,
                        choices=["no", "fp16", "bf16"])

    # Hub
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)

    args = parser.parse_args()
    return args


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "Cannot use both --report_to=wandb and --hub_token."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    ds_plugin = DeepSpeedPlugin(
        zero_stage=2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=args.max_grad_norm,
    )
    ds_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
        deepspeed_plugin=ds_plugin,
    )

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb: pip install wandb")

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

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # ── Load models ──
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler",
    )
    noise_scheduler_train = copy.deepcopy(noise_scheduler)

    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_3",
        revision=args.revision,
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision,
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision,
        subfolder="text_encoder_2",
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision,
        subfolder="text_encoder_3",
    )

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2",
        revision=args.revision, variant=args.variant,
    )
    text_encoder_three = text_encoder_cls_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3",
        revision=args.revision, variant=args.variant,
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant,
    )

    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        revision=args.revision, variant=args.variant,
    )

    # Freeze everything except transformer
    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Register hooks for checkpoint saving/loading
    accelerator.register_save_state_pre_hook(make_save_model_hook(accelerator))
    accelerator.register_load_state_pre_hook(make_load_model_hook(accelerator))

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # ── Optimizer ──
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("pip install bitsandbytes") from exc
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        [{"params": transformer.parameters(), "lr": args.learning_rate}],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ── Data ──
    train_dataloader = build_bucket_dataloader(
        jsonl_path=args.jsonl_path,
        image_dir=args.image_dir,
        batch_size=args.train_batch_size,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        num_workers=args.dataloader_num_workers,
        random_flip=args.random_flip,
        seed=args.seed or 0,
    )

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    def compute_text_embeddings(prompt):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length,
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds

    uncond_hidden_states, uncond_pooled_embeds = compute_text_embeddings("")

    logger.info(
        f"SOAR: lambda_aux={args.lambda_aux}, "
        f"num_rollout_paths={args.num_rollout_paths}, "
        f"trajectory_length={args.trajectory_length}, "
        f"num_sampling_steps={args.num_sampling_steps}, "
        f"sde_rollout_type={args.sde_rollout_type}, "
        f"sde_noise_scale={args.sde_noise_scale}, "
        f"cfg_scale_sampling={args.cfg_scale_sampling}, "
        f"cond_dropout_rate={args.cond_dropout_rate}, "
        f"sigma_upper_ratio={args.sigma_upper_ratio}"
    )

    # ── LR scheduler ──
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, lr_scheduler = accelerator.prepare(
        transformer, optimizer, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("soar-sd3", config=vars(args))

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Running SOAR training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' not found. "
                "Starting a new training run."
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
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # ── Training loop ──
    for epoch in range(first_epoch, args.num_train_epochs):
        if hasattr(train_dataloader, "batch_sampler") and hasattr(
            train_dataloader.batch_sampler, "set_epoch"
        ):
            train_dataloader.batch_sampler.set_epoch(epoch)

        transformer.train()

        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Encode images through VAE
                pixel_values = batch["pixel_values"].to(
                    device=accelerator.device, dtype=vae.dtype
                )
                with torch.no_grad():
                    z_0 = vae.encode(pixel_values).latent_dist.sample()
                    z_0 = (
                        z_0 - vae.config.shift_factor
                    ) * vae.config.scaling_factor
                    z_0 = z_0.to(dtype=weight_dtype)

                # Encode text
                prompts = batch["prompts"]
                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                    prompts
                )

                bsz = z_0.shape[0]

                # Conditional dropout for CFG
                if random.random() < args.cond_dropout_rate:
                    prompt_embeds = uncond_hidden_states.expand(bsz, -1, -1).clone()
                    pooled_prompt_embeds = uncond_pooled_embeds.expand(bsz, -1).clone()

                # Sample noise and timesteps
                z_1 = torch.randn_like(z_0)
                t0 = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                ).to(z_0.device)

                sigma_t0_1d, timesteps = t_to_sigma_timestep(
                    t0, noise_scheduler_train
                )
                sigma_t0 = sigma_t0_1d.view(bsz, 1, 1, 1).to(dtype=weight_dtype)

                # Noisy input: z_sigma = (1-sigma)*z_0 + sigma*z_1
                z_sigma_t0 = (1.0 - sigma_t0) * z_0 + sigma_t0 * z_1

                # Main forward pass
                v_t0 = transformer(
                    hidden_states=z_sigma_t0,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                # Main flow-matching loss
                v_gt = z_1 - z_0
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigma_t0,
                )
                loss_main = torch.mean(
                    (weighting.float() * (v_t0.float() - v_gt.float()) ** 2).reshape(
                        bsz, -1
                    ),
                    1,
                ).sum()

                # ── Auxiliary SOAR loss ──
                loss_aux = torch.tensor(0.0, device=z_0.device, dtype=torch.float32)
                local_main_count = torch.tensor(
                    float(bsz), device=z_0.device, dtype=torch.float32
                )
                local_aux_count = torch.tensor(
                    0.0, device=z_0.device, dtype=torch.float32
                )

                if args.lambda_aux > 0 and args.trajectory_length > 0:
                    rollout_model = unwrap_model(transformer, accelerator)

                    # CFG velocity for rollout
                    rollout_uncond = uncond_hidden_states.expand(bsz, -1, -1)
                    rollout_uncond_p = uncond_pooled_embeds.expand(bsz, -1)
                    rollout_hs = torch.cat(
                        [rollout_uncond.detach(), prompt_embeds.detach()], dim=0
                    )
                    rollout_pp = torch.cat(
                        [rollout_uncond_p.detach(), pooled_prompt_embeds.detach()],
                        dim=0,
                    )
                    rollout_latents = torch.cat(
                        [z_sigma_t0.detach(), z_sigma_t0.detach()], dim=0
                    )
                    rollout_ts = torch.cat([timesteps, timesteps], dim=0)

                    rollout_output = rollout_model(
                        hidden_states=rollout_latents,
                        timestep=rollout_ts,
                        encoder_hidden_states=rollout_hs,
                        pooled_projections=rollout_pp,
                        return_dict=False,
                    )[0]
                    v_uncond, v_cond = rollout_output.chunk(2)
                    v_cfg = (
                        v_uncond + args.cfg_scale_sampling * (v_cond - v_uncond)
                    ).detach()

                    # Generate auxiliary supervision points
                    aux_points = single_step_aux_points(
                        z_t0=z_sigma_t0.detach(),
                        t0=t0.detach(),
                        v_cfg=v_cfg,
                        z_1=z_1.detach(),
                        num_paths=args.num_rollout_paths,
                        points_per_path=args.trajectory_length,
                        sigma_upper_ratio=args.sigma_upper_ratio,
                        noise_scheduler=noise_scheduler_train,
                        num_sampling_steps=args.num_sampling_steps,
                        sde_rollout_type=args.sde_rollout_type,
                        sde_noise_scale=args.sde_noise_scale,
                    )

                    for point in aux_points:
                        si = point["sample_indices"]
                        z_t_prime = point["latents"].to(dtype=weight_dtype)
                        sigma_t_prime = point["sigmas"].view(-1, 1, 1, 1).to(
                            dtype=weight_dtype
                        )
                        ts_t_prime = point["timesteps"]
                        z_0_sub = z_0[si]
                        pe_sub = prompt_embeds[si]
                        ppe_sub = pooled_prompt_embeds[si]

                        # Corrected target: v_corr = (z_t' - z_0) / sigma_t'
                        v_corr = (z_t_prime - z_0_sub) / sigma_t_prime
                        v_pred = rollout_model(
                            hidden_states=z_t_prime,
                            timestep=ts_t_prime,
                            encoder_hidden_states=pe_sub,
                            pooled_projections=ppe_sub,
                            return_dict=False,
                        )[0]

                        w_pt = compute_loss_weighting_for_sd3(
                            weighting_scheme=args.weighting_scheme,
                            sigmas=sigma_t_prime,
                        )
                        per_sample = torch.mean(
                            (w_pt.float() * (v_pred.float() - v_corr.float()) ** 2).reshape(
                                si.shape[0], -1
                            ),
                            1,
                        )
                        loss_aux += args.lambda_aux * per_sample.sum()
                        local_aux_count += float(si.shape[0])

                # Normalize by global count
                global_main_count = local_main_count.detach().clone()
                global_aux_count = local_aux_count.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(global_main_count, op=dist.ReduceOp.SUM)
                    dist.all_reduce(global_aux_count, op=dist.ReduceOp.SUM)

                total_count = torch.clamp(
                    global_main_count + args.lambda_aux * global_aux_count, min=1.0
                )
                loss_total = (
                    (loss_main + loss_aux) / total_count * accelerator.num_processes
                )

                accelerator.backward(loss_total)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            aux_per_sample = 0.0
            if global_main_count.item() > 0:
                aux_per_sample = (
                    global_aux_count / global_main_count
                ).detach().item()

            logs = {
                "loss": loss_total.detach().item(),
                "loss_main": (
                    loss_main / torch.clamp(global_main_count, min=1.0)
                ).detach().item(),
                "loss_aux": (
                    (loss_aux / global_aux_count).detach().item()
                    if global_aux_count.item() > 0
                    else 0.0
                ),
                "aux_points_per_sample": aux_per_sample,
                "lr": lr_scheduler.get_last_lr()[0],
            }

            if accelerator.sync_gradients:
                progress_bar.set_postfix(**logs)
                progress_bar.update(1)
                global_step += 1

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    if (
                        accelerator.is_main_process
                        and args.checkpoints_total_limit is not None
                    ):
                        ckpts = os.listdir(args.output_dir)
                        ckpts = [d for d in ckpts if d.startswith("checkpoint")]
                        ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
                        if len(ckpts) >= args.checkpoints_total_limit:
                            for rm in ckpts[: len(ckpts) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, rm))

                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        os.makedirs(save_path, exist_ok=True)
                        unwrap_model(transformer, accelerator).save_pretrained(
                            os.path.join(save_path, "transformer")
                        )
                        logger.info(f"Saved transformer to {save_path}")
                    accelerator.wait_for_everyone()

            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # ── Save final model ──
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrap_model(transformer, accelerator).save_pretrained(
            os.path.join(args.output_dir, "transformer")
        )
        logger.info(f"Saved final transformer to {args.output_dir}/transformer")

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
