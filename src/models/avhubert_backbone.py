from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn

from src.utils.avhubert_env import import_avhubert_modules


class AVHubertVideoBackbone(nn.Module):
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

        self.model, self.load_info = self._load_avhubert_model()
        self.output_dim = self._infer_output_dim(self.model)
        self.model.float()
        self.model.modality_dropout = 0.0
        self.model.audio_dropout = 0.0

        if self.freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()

    def _load_avhubert_model(self):
        fairseq, _, _, _ = import_avhubert_modules(self.avhubert_repo)
        from fairseq import checkpoint_utils, tasks  # type: ignore
        from fairseq.dataclass.utils import convert_namespace_to_omegaconf  # type: ignore

        state = checkpoint_utils.load_checkpoint_to_cpu(str(self.checkpoint_path))
        cfg = state.get("cfg")
        if cfg is None:
            cfg = convert_namespace_to_omegaconf(state["args"])

        model_state = state["model"]
        is_finetuned_seq2seq = any(key.startswith("encoder.w2v_model.") for key in model_state)

        if is_finetuned_seq2seq:
            model_cfg = getattr(cfg, "model", None)
            w2v_args = getattr(model_cfg, "w2v_args", None)
            if w2v_args is None:
                raise RuntimeError(
                    "The checkpoint looks like a fine-tuned seq2seq AV-HuBERT model, "
                    "but cfg.model.w2v_args is missing."
                )
            if isinstance(w2v_args, Namespace):
                w2v_args = convert_namespace_to_omegaconf(w2v_args)

            task = tasks.setup_task(w2v_args.task)
            model = task.build_model(w2v_args.model)
            prefix = "encoder.w2v_model."
            encoder_state = {
                key[len(prefix) :]: value
                for key, value in model_state.items()
                if key.startswith(prefix)
            }
            incompatible = model.load_state_dict(encoder_state, strict=False)
        else:
            task = tasks.setup_task(cfg.task)
            model = task.build_model(cfg.model)
            incompatible = model.load_state_dict(model_state, strict=False)

        if hasattr(model, "remove_pretraining_modules"):
            model.remove_pretraining_modules()

        load_info = {
            "checkpoint_path": str(self.checkpoint_path),
            "is_finetuned_seq2seq": is_finetuned_seq2seq,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
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
        video: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        source = {"audio": None, "video": video}

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
