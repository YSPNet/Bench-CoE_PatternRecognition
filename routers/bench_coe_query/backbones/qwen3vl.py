"""Frozen Qwen3-VL native-chat query feature adapter."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ..assets import StrictAssetResolver
from ..config import ExperimentContract, effective_model_path, sha256_file
from ..input_content import feature_input_content_fields
from ..schemas import StrictSample
from .base import BackboneFeatureExtractor, FeatureBatchOutput
from .utils import force_offline_environment, freeze_and_verify, parameter_summary_checksum, require_model_identity


class Qwen3VLFeatureExtractor(BackboneFeatureExtractor):
    name = "qwen3vl"

    def __init__(
        self,
        contract: ExperimentContract,
        *,
        device: str | torch.device,
        dataset_root: str | Path | None = None,
    ):
        force_offline_environment()
        self.contract = contract
        self.model_path = effective_model_path(contract, "qwen3vl")
        config = require_model_identity(
            self.model_path,
            expected_model_type="qwen3_vl",
            expected_architecture="Qwen3VLForConditionalGeneration",
        )
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=True
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=True,
            dtype=self.dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        freeze_and_verify(self.model)
        self.feature_dim = int(config["text_config"]["hidden_size"])
        self.resolver = StrictAssetResolver(dataset_root or contract.dataset_root)

    @torch.inference_mode()
    def encode_batch(self, sample_batch: Sequence[StrictSample]) -> FeatureBatchOutput:
        if not sample_batch:
            raise ValueError("Qwen3-VL encode_batch requires at least one sample")
        features = []
        metadata = []
        preprocess_seconds = 0.0
        forward_seconds = 0.0
        total_sequence_tokens = 0
        total_visual_tokens = 0
        for sample in sample_batch:
            preprocess_started = time.perf_counter()
            resolved = self.resolver.resolve_sample(sample)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": resolved.image},
                        {"type": "text", "text": sample.text},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            preprocess_seconds += time.perf_counter() - preprocess_started
            forward_started = time.perf_counter()
            output = self.model.model(**inputs, return_dict=True)
            attention_mask = inputs["attention_mask"]
            weights = attention_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
            pooled = (output.last_hidden_state * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            if pooled.shape != (1, self.feature_dim):
                raise RuntimeError(f"unexpected Qwen3-VL pooled shape: {tuple(pooled.shape)}")
            features.append(pooled.float().cpu())
            forward_seconds += time.perf_counter() - forward_started
            visual_tokens = None
            if "image_grid_thw" in inputs:
                grid = inputs["image_grid_thw"]
                visual_tokens = int(grid.prod(dim=-1).sum().item() // 4)
            sequence_tokens = int(attention_mask.sum().item())
            total_sequence_tokens += sequence_tokens
            total_visual_tokens += int(visual_tokens or 0)
            metadata.append(
                {
                    "status": "ok",
                    "sample_id": sample.sample_id,
                    "sequence_tokens": sequence_tokens,
                    "visual_tokens": visual_tokens,
                    "pooling_positions": int(attention_mask.sum().item()),
                    "original_image_size": list(resolved.original_size),
                    "image_grid_thw": inputs.get("image_grid_thw").detach().cpu().tolist()
                    if "image_grid_thw" in inputs
                    else None,
                }
            )
        self.last_batch_timing = {
            "preprocess_seconds": preprocess_seconds,
            "forward_seconds": forward_seconds,
            "text_tokens": total_sequence_tokens,
            "visual_tokens": total_visual_tokens,
        }
        return FeatureBatchOutput(features=torch.cat(features, dim=0), metadata=tuple(metadata))

    def feature_signature_payload(self) -> Mapping[str, Any]:
        model = self.contract.values["models"]["qwen3vl"]
        return {
            "schema_version": 2,
            "backbone": self.name,
            "identity": model["identity"],
            "revision": model["revision"],
            "config_sha256": model["config_sha256"],
            "tokenizer_config_sha256": model["tokenizer_config_sha256"],
            "preprocessor_config_sha256": model["preprocessor_config_sha256"],
            "chat_template_sha256": model["chat_template_sha256"],
            "modeling_source_sha256": model["modeling_source_sha256"],
            "processing_source_sha256": model["processing_source_sha256"],
            "processor_min_pixels": model["processor_min_pixels"],
            "processor_max_pixels": model["processor_max_pixels"],
            "prompt_content": "native_user_message_with_exactly_one_RGB_image_and_sample_text",
            "add_generation_prompt": True,
            "pooling": model["pooling"],
            "dtype": model["dtype"],
            "attention": model["attention"],
            "feature_dim": self.feature_dim,
            "input_content": feature_input_content_fields(self.contract, self.name),
            "adapter_code_sha256": sha256_file(Path(__file__)),
            "contract_sha256": self.contract.sha256,
        }

    def parameter_checksum(self) -> str:
        return parameter_summary_checksum(self.model)
