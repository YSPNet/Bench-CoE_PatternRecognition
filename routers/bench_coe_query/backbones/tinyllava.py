"""Frozen TinyLLaVA two-branch query feature adapter."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..assets import StrictAssetResolver
from ..config import ExperimentContract, effective_model_path, sha256_file
from ..input_content import feature_input_content_fields
from ..schemas import StrictSample
from .base import BackboneFeatureExtractor, FeatureBatchOutput
from .utils import force_offline_environment, freeze_and_verify, parameter_summary_checksum, require_model_identity


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def format_tinyllava_text(text: str) -> str:
    cleaned = text.replace("<image 1>", "").replace("<image>", "").strip()
    return f"Question:\n{cleaned}"


class TinyLLaVAFeatureExtractor(BackboneFeatureExtractor):
    name = "tinyllava"

    def __init__(
        self,
        contract: ExperimentContract,
        *,
        device: str | torch.device,
        dataset_root: str | Path | None = None,
    ):
        force_offline_environment()
        self.contract = contract
        self.model_path = effective_model_path(contract, "tinyllava")
        config = require_model_identity(
            self.model_path,
            expected_model_type="tinyllava",
            expected_architecture="TinyLlavaForConditionalGeneration",
        )
        self.device = torch.device(device)
        self.dtype = torch.float16
        self.max_text_length = 256
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        freeze_and_verify(self.model)
        self.image_processor = self.model.vision_tower._image_processor
        self.image_size = int(config["vision_config"]["image_size"])
        self.feature_dim = int(config["hidden_size"]) * 2
        self.resolver = StrictAssetResolver(dataset_root or contract.dataset_root)

    @torch.inference_mode()
    def encode_batch(self, sample_batch: Sequence[StrictSample]) -> FeatureBatchOutput:
        if not sample_batch:
            raise ValueError("TinyLLaVA encode_batch requires at least one sample")
        preprocess_started = time.perf_counter()
        resolved = [self.resolver.resolve_sample(sample) for sample in sample_batch]
        images = [item.image for item in resolved]
        processed = self.image_processor(images=images, return_tensors="pt")
        pixels = processed["pixel_values"].to(device=self.device, dtype=self.dtype)
        encoded = self.tokenizer(
            [format_tinyllava_text(sample.text) for sample in sample_batch],
            truncation=True,
            padding=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        preprocess_seconds = time.perf_counter() - preprocess_started
        forward_started = time.perf_counter()
        with torch.autocast(
            device_type="cuda",
            dtype=self.dtype,
            enabled=self.device.type == "cuda",
        ):
            image_tokens = self.model.encode_images(pixels)
            vision = image_tokens.mean(dim=1)
            text_output = self.model.language_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            text = masked_mean(text_output.last_hidden_state, attention_mask)
        features = torch.cat((vision.float(), text.float()), dim=-1)
        if features.shape != (len(sample_batch), self.feature_dim):
            raise RuntimeError(
                f"unexpected TinyLLaVA feature shape {tuple(features.shape)}, expected {(len(sample_batch), self.feature_dim)}"
            )
        features = features.cpu()
        forward_seconds = time.perf_counter() - forward_started
        self.last_batch_timing = {
            "preprocess_seconds": preprocess_seconds,
            "forward_seconds": forward_seconds,
            "text_tokens": int(attention_mask.sum().item()),
            "visual_tokens": int(image_tokens.shape[1]) * len(sample_batch),
        }
        metadata = tuple(
            {
                "status": "ok",
                "sample_id": sample.sample_id,
                "text_tokens": int(attention_mask[position].sum().item()),
                "visual_tokens": int(image_tokens.shape[1]),
                "vision_pooling_positions": int(image_tokens.shape[1]),
                "text_pooling_positions": int(attention_mask[position].sum().item()),
                "original_image_size": list(resolved[position].original_size),
                "processor_image_size": self.image_size,
            }
            for position, sample in enumerate(sample_batch)
        )
        return FeatureBatchOutput(features=features, metadata=metadata)

    def feature_signature_payload(self) -> Mapping[str, Any]:
        model = self.contract.values["models"]["tinyllava"]
        return {
            "schema_version": 2,
            "backbone": self.name,
            "identity": model["identity"],
            "revision": model["revision"],
            "config_sha256": model["config_sha256"],
            "remote_code_sha256": model["remote_code_sha256"],
            "tokenizer_config_sha256": model["tokenizer_config_sha256"],
            "processor": model["processor"],
            "processor_image_size": self.image_size,
            "prompt_template": model["prompt_template"],
            "pooling": model["pooling"],
            "hidden_state_layer": model["hidden_state_layer"],
            "max_text_length": self.max_text_length,
            "dtype": model["dtype"],
            "attention": model["attention"],
            "feature_dim": self.feature_dim,
            "input_content": feature_input_content_fields(self.contract, self.name),
            "adapter_code_sha256": sha256_file(Path(__file__)),
            "contract_sha256": self.contract.sha256,
        }

    def parameter_checksum(self) -> str:
        return parameter_summary_checksum(self.model)
