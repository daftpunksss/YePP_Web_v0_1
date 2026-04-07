from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

from predictor_web import config


class LightAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int = 256, kernel_size: int = 9):
        super().__init__()
        padding = kernel_size // 2
        self.conv_e = nn.Conv1d(d_in, d_out, kernel_size, padding=padding, bias=True)
        self.conv_v = nn.Conv1d(d_in, d_out, kernel_size, padding=padding, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        e = self.conv_e(x)
        v = self.conv_v(x)

        mask = mask.unsqueeze(1).bool()
        e = e.masked_fill(~mask, -1e9)

        attn = torch.softmax(e, dim=-1)
        x_prime = (attn * v).sum(dim=-1)

        v_masked = v.masked_fill(~mask, -1e9)
        v_max = v_masked.max(dim=-1).values

        return torch.cat([x_prime, v_max], dim=-1)


class SpeciesLMLightAttention(nn.Module):
    def __init__(self, attn_out: int = 256, hidden_dim: int = 512, kernel_size: int = 9):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
        self.backbone = AutoModelForMaskedLM.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)

        for p in self.backbone.parameters():
            p.requires_grad = False

        for p in self.backbone.bert.encoder.layer[-2:].parameters():
            p.requires_grad = True

        d_in = self.backbone.config.hidden_size
        self.attn = LightAttention(d_in=d_in, d_out=attn_out, kernel_size=kernel_size)

        mlp_in = attn_out * 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = outputs.hidden_states[-1]
        seq_emb = self.attn(last_hidden, attention_mask)
        return self.mlp(seq_emb).squeeze(-1)


def load_checkpoint(model: SpeciesLMLightAttention, checkpoint_path: str, device: torch.device) -> SpeciesLMLightAttention:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(path, map_location=device)
    state_dict_raw = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict_raw.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model
