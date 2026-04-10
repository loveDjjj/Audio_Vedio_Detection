from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.utils.avhubert_env import import_avhubert_modules


def _to_plain_dict(config: Any) -> dict:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    if isinstance(config, dict):
        return config
    raise TypeError(f"Unsupported checkpoint config type: {type(config)!r}")


def _merge_with_schema_defaults(schema: Any, overrides: dict) -> DictConfig:
    defaults = OmegaConf.to_container(OmegaConf.structured(schema), resolve=False)
    return OmegaConf.merge(OmegaConf.create(defaults), OmegaConf.create(overrides))


def _resolve_checkpoint_configs(state: dict) -> dict:
    cfg = state.get("cfg")
    if cfg is None:
        raise RuntimeError("Checkpoint is missing `cfg`; unsupported AV-HuBERT checkpoint format.")

    cfg_dict = _to_plain_dict(cfg)
    model_state = state["model"]
    is_seq2seq = any(key.startswith("encoder.w2v_model.") for key in model_state)

    if is_seq2seq:
        w2v_args = cfg_dict["model"].get("w2v_args")
        if not isinstance(w2v_args, dict):
            raise RuntimeError("Seq2seq checkpoint is missing `cfg.model.w2v_args`.")
        model_cfg = w2v_args["model"]
        task_cfg = w2v_args["task"]
        checkpoint_prefix = "encoder.w2v_model."
    else:
        model_cfg = cfg_dict["model"]
        task_cfg = cfg_dict["task"]
        checkpoint_prefix = ""

    return {
        "checkpoint_prefix": checkpoint_prefix,
        "is_seq2seq": is_seq2seq,
        "model_cfg": _to_plain_dict(model_cfg),
        "task_cfg": _to_plain_dict(task_cfg),
    }


def load_avhubert_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    state = torch.load(str(checkpoint_path), map_location="cpu")
    resolved = _resolve_checkpoint_configs(state)
    model_cfg = resolved["model_cfg"]
    task_cfg = resolved["task_cfg"]

    return {
        "checkpoint_prefix": resolved["checkpoint_prefix"],
        "is_seq2seq": resolved["is_seq2seq"],
        "encoder_embed_dim": int(model_cfg["encoder_embed_dim"]),
        "audio_feat_dim": int(model_cfg["audio_feat_dim"]),
        "modality_fuse": model_cfg.get("modality_fuse"),
        "stack_order_audio": int(task_cfg.get("stack_order_audio", 1)),
        "audio_normalize": bool(task_cfg.get("normalize", False)),
        "modalities": list(task_cfg.get("modalities", ["audio", "video"])),
    }


class AVHubertBackbone(nn.Module):
    def __init__(
        self,
        checkpoint_path: Path,
        avhubert_repo: Path,
        freeze: bool = True,
        output_layer: int | None = None,
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.avhubert_repo = avhubert_repo
        self.freeze = freeze
        self.output_layer = output_layer

        self.metadata = load_avhubert_checkpoint_metadata(checkpoint_path)
        self.model, self.load_info = self._load_avhubert_model()
        self.output_dim = self._infer_output_dim(self.model)
        self.model.float()
        self.model.modality_dropout = 0.0
        self.model.audio_dropout = 0.0

        if self.freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()

    def _load_avhubert_model(self) -> tuple[nn.Module, dict[str, Any]]:
        _, hubert_pretraining_module, hubert_module, _ = import_avhubert_modules(self.avhubert_repo)

        state = torch.load(str(self.checkpoint_path), map_location="cpu")
        resolved = _resolve_checkpoint_configs(state)
        task_cfg = _merge_with_schema_defaults(
            hubert_pretraining_module.AVHubertPretrainingConfig,
            resolved["task_cfg"],
        )
        model_cfg = _merge_with_schema_defaults(
            hubert_module.AVHubertConfig,
            resolved["model_cfg"],
        )
        if "input_modality" in resolved["task_cfg"]:
            model_cfg.input_modality = resolved["task_cfg"]["input_modality"]
        model = hubert_module.AVHubertModel(model_cfg, task_cfg, dictionaries=[None])

        checkpoint_prefix = resolved["checkpoint_prefix"]
        model_state = state["model"]
        if checkpoint_prefix:
            backbone_state = {
                key[len(checkpoint_prefix) :]: value
                for key, value in model_state.items()
                if key.startswith(checkpoint_prefix)
            }
        else:
            backbone_state = model_state

        incompatible = model.load_state_dict(backbone_state, strict=False)
        if hasattr(model, "remove_pretraining_modules"):
            model.remove_pretraining_modules()

        load_info = {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_prefix": checkpoint_prefix,
            "is_seq2seq": resolved["is_seq2seq"],
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "encoder_embed_dim": int(self.metadata["encoder_embed_dim"]),
        }
        return model, load_info

    @staticmethod
    def _infer_output_dim(model: nn.Module) -> int:
        if hasattr(model, "encoder_embed_dim"):
            return int(model.encoder_embed_dim)
        if hasattr(model, "encoder") and hasattr(model.encoder, "embedding_dim"):
            return int(model.encoder.embedding_dim)
        raise RuntimeError("Unable to infer AV-HuBERT output dimension from the loaded backbone.")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(
        self,
        audio: torch.Tensor | None,
        video: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        source = {"audio": audio, "video": video}

        if self.freeze:
            with torch.no_grad():
                features, feature_padding_mask = self.model.extract_finetune(
                    source=source,
                    padding_mask=padding_mask,
                    output_layer=self.output_layer,
                )
        else:
            features, feature_padding_mask = self.model.extract_finetune(
                source=source,
                padding_mask=padding_mask,
                output_layer=self.output_layer,
            )

        return features, feature_padding_mask


AVHubertVideoBackbone = AVHubertBackbone
