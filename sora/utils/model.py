"""
Text encoding and model utility functions for SD3.5 training.

Provides prompt encoding with CLIP + T5 text encoders and
model save/load hook helpers for Accelerate checkpointing.
"""

import os

import torch
from transformers import PretrainedConfig

from diffusers import SD3Transformer2DModel
from diffusers.utils.torch_utils import is_compiled_module


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    revision: str,
    subfolder: str = "text_encoder",
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    text_inputs = tokenizer(
        prompt, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    )
    prompt_embeds = text_encoder(
        text_inputs.input_ids.to(device), output_hidden_states=True
    )
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(
        batch_size * num_images_per_prompt, seq_len, -1
    )
    return prompt_embeds, pooled_prompt_embeds


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    text_inputs = tokenizer(
        prompt, padding="max_length", max_length=max_sequence_length,
        truncation=True, add_special_tokens=True, return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(
        batch_size * num_images_per_prompt, seq_len, -1
    )
    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    """Encode a prompt using CLIP (x2) + T5 text encoders."""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        pe, ppe = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(pe)
        clip_pooled_prompt_embeds_list.append(ppe)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    return prompt_embeds, pooled_prompt_embeds


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def make_save_model_hook(accelerator):
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(
                    unwrap_model(model, accelerator), SD3Transformer2DModel
                ):
                    unwrap_model(model, accelerator).save_pretrained(
                        os.path.join(output_dir, "transformer")
                    )
                weights.pop()
    return save_model_hook


def make_load_model_hook(accelerator):
    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            model = models.pop()
            if isinstance(
                unwrap_model(model, accelerator), SD3Transformer2DModel
            ):
                load_model = SD3Transformer2DModel.from_pretrained(
                    input_dir, subfolder="transformer"
                )
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model
    return load_model_hook
