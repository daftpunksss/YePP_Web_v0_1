import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


class SpeciesLMLinearRegressor(nn.Module):
    """
    使用 SpeciesLM (DNABERT2) 提取序列 embedding 进行回归。
    只微调最后一个 Transformer Block + 自定义 MLP 头。
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # ---------- 1. 载入 tokenizer & 预训练模型 ----------
        model_id  = "gagneurlab/SpeciesLM"
        revision  = "upstream_species_lm"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.backbone  = AutoModelForMaskedLM.from_pretrained(model_id, revision=revision)

        # ---------- 2. 冻结全部参数 ----------
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ---------- 3. 解冻最后一个 Transformer block ----------
        # for p in self.backbone.bert.encoder.layer[-1].parameters():
        #     p.requires_grad = True

        # 也可以解冻 Embedding 中的 species token（可选）
        # species_token_id = self.tokenizer.convert_tokens_to_ids("[candida_glabrata]")
        # self.backbone.bert.embeddings.word_embeddings.weight[species_token_id].requires_grad = True

        # ---------- 4. 构建 MLP 头 ----------
        input_dim = self.backbone.config.hidden_size  # 768
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, 1)
        )

    # ---------- 前向传播 ----------
    def forward(self, input_ids, attention_mask=None):
        """
        input_ids      : [B, L]
        attention_mask : [B, L]  (若 None, 自动全部为 1)
        返回           : [B]  (回归预测)
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # 取所有 hidden_states
        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True,
                                return_dict=True)

        last_hidden = outputs.hidden_states[-1]        # [B, L, H]

        # ---- mean pooling（只对有效 token）----
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())
        sum_emb = (last_hidden * mask_expanded).sum(dim=1)      # [B, H]
        len_emb = mask_expanded.sum(dim=1) + 1e-8
        seq_emb = sum_emb / len_emb                             # [B, H]

        # ---- MLP 回归 ----
        preds = self.mlp(seq_emb).squeeze(-1)                   # [B]

        return preds
    
# ---------------------- LightAttention ----------------------
class LightAttention(nn.Module):
    """
    输入  : x       -> [B, L, d_in]
           mask    -> [B, L]   (1=valid, 0=pad)
    输出  : out     -> [B, 2*d_out]   (concat{x' , v_max})
    """
    def __init__(self, d_in: int, d_out: int = 256, kernel_size: int = 9):
        super().__init__()
        pad = kernel_size // 2
        self.conv_e = nn.Conv1d(d_in, d_out, kernel_size, padding=pad, bias=True)
        self.conv_v = nn.Conv1d(d_in, d_out, kernel_size, padding=pad, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # [B, L, d_in] -> [B, d_in, L]
        x = x.permute(0, 2, 1)

        e = self.conv_e(x)                    # [B, d_out, L]
        v = self.conv_v(x)                    # [B, d_out, L]

        # ---- mask padding token ----
        mask = mask.unsqueeze(1).bool()       # [B, 1, L]
        e = e.masked_fill(~mask, -1e9)

        α = torch.softmax(e, dim=-1)          # attention weights  [B, d_out, L]

        # ---- 加权和 x' ----
        x_prime = (α * v).sum(dim=-1)         # [B, d_out]

        # ---- max-pool v ----
        v_mask = v.masked_fill(~mask, -1e9)
        v_max  = v_mask.max(dim=-1).values    # [B, d_out]

        return torch.cat([x_prime, v_max], dim=-1)   # [B, 2*d_out]
    

class SpeciesLMLightAttention(nn.Module):
    """
    使用 SpeciesLM (DNABERT2) 提取序列 embedding 进行回归。
    只微调最后一个 Transformer Block + 自定义 MLP 头。
    """

    def __init__(self, 
                 attn_out: int = 256,          # d_out of LA
                 hidden_dim: int = 512,        # 随后 MLP 的隐藏层维度
                 kernel_size: int = 9):
        super().__init__()

        # ---------- 1. 载入 tokenizer & 预训练模型 ----------
        model_id  = "gagneurlab/SpeciesLM"
        revision  = "upstream_species_lm"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.backbone  = AutoModelForMaskedLM.from_pretrained(model_id, revision=revision)

        # ---------- 2. 冻结全部参数 ----------
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ---------- 3. 解冻最后一个 Transformer block ----------
        for p in self.backbone.bert.encoder.layer[-2:].parameters():
            p.requires_grad = True
            
        d_in = self.backbone.config.hidden_size       # 768

        # 也可以解冻 Embedding 中的 species token（可选）
        
        # ---- 4. 仅解冻指定 species token 的 embedding ----
        # sp_token = "kazachstania_africana_cbs_2517_gca_000304475"                     # ← 你的物种 token
        # # yarrowia_lipolytica, yarrowia_lipolytica_gca_001761485, yarrowia_lipolytica_gca_003367845, yarrowia_lipolytica_gca_014490615
        # # kluyveromyces_marxianus, kluyveromyces_marxianus_dmku3_1042_gca_001417885, kluyveromyces_marxianus_gca_001417835
        # # komagataella_pastoris, komagataella_pastoris_gca_001708105, komagataella_phaffii_cbs_7435_gca_000223565, komagataella_phaffii_gs115_gca_001746955
        # # kazachstania_africana_cbs_2517_gca_000304475 for S. cerevisiae
        # self.sp_id = self.tokenizer.convert_tokens_to_ids(sp_token)
        # vec0 = self.backbone.bert.embeddings.word_embeddings.weight[self.sp_id].detach().clone()
        # self.species_embed = nn.Parameter(vec0)
        
        
        # --- 4. LightAttention ---
        self.attn = LightAttention(d_in=d_in,
                                   d_out=attn_out,
                                   kernel_size=kernel_size)     # 输出 2*attn_out
        

        # ---------- 4. 构建 MLP 头 ----------
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
            nn.Linear(hidden_dim // 4, 1)
        )

    # ---------- 前向传播 ----------
    def forward(self, input_ids, attention_mask=None):
        """
        input_ids      : [B, L]
        attention_mask : [B, L]  (若 None, 自动全部为 1)
        返回           : [B]  (回归预测)
        """
        if attention_mask is None:
            print("Warning: attention_mask is None, using default mask.")
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        #  # ① 原始词向量
        # embeds = self.backbone.bert.embeddings.word_embeddings(input_ids)

        # print(self.backbone.bert.embeddings.word_embeddings.weight.requires_grad)
        
        # # ② 把 sp_id 位置的行替换成可训练向量
        # mask = (input_ids == self.sp_id).unsqueeze(-1)          # [B,L,1] bool
        # sp_vec = self.species_embed.unsqueeze(0).unsqueeze(0)   # [1,1,H]
        # embeds = torch.where(mask, sp_vec, embeds)

        # 取所有 hidden_states
        outputs = self.backbone(
                                input_ids=input_ids,
                                # inputs_embeds=embeds,
                                attention_mask=attention_mask,
                                output_hidden_states=True,
                                return_dict=True
                                )

        last_hidden = outputs.hidden_states[-1]        # [B, L, H]

        # ---- LightAttention ----
        seq_emb = self.attn(last_hidden, attention_mask)
        
        # ---- MLP 回归 ----
        preds = self.mlp(seq_emb).squeeze(-1)                   # [B]

        return preds

