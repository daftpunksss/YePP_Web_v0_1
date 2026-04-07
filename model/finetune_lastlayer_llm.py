import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

class NTRegressor(nn.Module):
    """
    从 Nucleotide-Transformer 提取序列嵌入并做回归。
    仅解冻最后一个 Transformer Block + lm_head。
    """
    def __init__(
        self,
        llm_model_path='/root/autodl-tmp/graph_model/yeast_sequence_processing/DNABERT_and_other_pretrain_model/NT100m',
        hidden_dim: int = 512,
        unfreeze_last_n: int = 1,          # 想解冻最后 N 层就改这里
    ):
        super().__init__()

        # --- 1. 载入 tokenizer / 预训练模型 ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_path, trust_remote_code=True
        )
        self.backbone = AutoModelForMaskedLM.from_pretrained(
            llm_model_path, trust_remote_code=True
        )

        # --- 2. 冻结全部，再解冻最后 n 层 + lm_head ---
        for p in self.backbone.parameters():
            p.requires_grad = False

        encoder_layers = (
            self.backbone.bert.encoder.layer   # DNABERT/NT 系列
            if hasattr(self.backbone, "bert")
            else self.backbone.esm.encoder.layer  # InstaDeep-ESM 系列
        )
        for block in encoder_layers[-unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad = True

        for p in self.backbone.lm_head.parameters():
            p.requires_grad = True

        # --- 3. 自己的回归头 ---
        input_dim = self.backbone.config.hidden_size * 4   # 768 × 4 = 3 072  (若你用 2560 就改这里)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, 1),
        )

    # ---------- 前向传播 ----------
    def forward(self, input_ids, attention_mask):
        """
        input_ids      : [B, L]
        attention_mask : [B, L]
        labels         : [B]  (可选)
        """
        outs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outs.hidden_states[-1]              # [B, L, H]

        # ====== 池化策略：mean + max + CLS + GAP（示例，任选其一） ======
        mean_pool = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) \
                    / attention_mask.sum(1, keepdim=True)               # [B, H]
        max_pool, _ = (last_hidden + (1 - attention_mask.unsqueeze(-1)) * -1e4).max(1)  # [B, H]
        cls_pool = last_hidden[:, 0]                                     # [B, H]
        gap_pool = last_hidden.mean(1)                                   # [B, H]

        concat = torch.cat([mean_pool, max_pool, cls_pool, gap_pool], dim=-1)  # [B, 4H]

        preds = self.mlp(concat).squeeze(-1)  # [B]

        return preds
