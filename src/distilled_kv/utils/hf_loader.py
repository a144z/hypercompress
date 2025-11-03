from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFLoaderConfig:
    model_name: str
    revision: Optional[str] = None
    dtype: str = "bfloat16"
    device_map: str | None = "auto"
    trust_remote_code: bool = False
    cache_dir: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_auth_token: Optional[str] = None


def load_causal_lm(config: HFLoaderConfig):
    """Load a Hugging Face causal LM with optional quantization."""

    dtype = getattr(torch, config.dtype)
    kwargs = {
        "dtype": dtype,
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
        "cache_dir": config.cache_dir,
        "use_auth_token": config.use_auth_token,
    }

    if config.device_map:
        kwargs["device_map"] = config.device_map

    quant_config = None
    if config.load_in_8bit or config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        kwargs["quantization_config"] = quant_config

    # Use 'token' parameter (newer API) instead of 'use_auth_token' if available
    token_kwargs = {}
    if config.use_auth_token:
        # Try 'token' first (transformers >= 4.21), fallback to 'use_auth_token'
        try:
            from transformers import __version__ as tf_version
            major, minor = map(int, tf_version.split(".")[:2])
            if major > 4 or (major == 4 and minor >= 21):
                token_kwargs["token"] = config.use_auth_token
            else:
                token_kwargs["use_auth_token"] = config.use_auth_token
        except Exception:
            token_kwargs["token"] = config.use_auth_token

    merged_kwargs = {k: v for k, v in {**kwargs, **token_kwargs}.items() if v is not None}

    try:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **merged_kwargs)
    except TypeError:
        # Fallback for older transformer versions that only accept torch_dtype
        merged_kwargs.pop("dtype", None)
        merged_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **merged_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.revision,
        use_fast=True,
        trust_remote_code=config.trust_remote_code,
        cache_dir=config.cache_dir,
        **token_kwargs,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


__all__ = ["HFLoaderConfig", "load_causal_lm"]


