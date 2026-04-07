from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class RegressionModelFreeze(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=512):
        super(RegressionModelFreeze, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/graph_model/yeast_sequence_processing/DNABERT_and_other_pretrain_model/NT2.5b")
        self.pretrained_model = AutoModelForMaskedLM.from_pretrained("/root/autodl-tmp/graph_model/yeast_sequence_processing/DNABERT_and_other_pretrain_model/NT2.5b")
        for name, param in self.pretrained_model.named_parameters():
            if 'encoder.layer' in name:
                parts = name.split('.')
                if len(parts) > 2 and parts[2].isdigit():  # 确保可以安全转换为整数
                    layer_number = int(parts[2])
                    param.requires_grad = layer_number >= 28  # 简化条件
                    print("-------------update several layers of NT2.5B------------------")
                else:
                    param.requires_grad = False  # 处理无效层号的情况
            else:
                param.requires_grad = False
        self.hidden_linear_1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.hidden_linear_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4)
        )
        self.out = nn.Linear(hidden_dim // 4, 1)

    def forward(self, tokens_ids):
        attention_mask = tokens_ids != self.tokenizer.pad_token_id
        torch_outs = self.pretrained_model(
            tokens_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )
        embeddings = torch_outs['hidden_states'][-1]
        print(f'embeddings shape:{embeddings.shape}')
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)
        mean_sequence_embeddings = torch.sum(attention_mask * embeddings, axis=-2) / torch.sum(attention_mask, axis=1)
        
        x = self.hidden_linear_1(mean_sequence_embeddings)
        x = self.hidden_linear_2(x)
        return self.out(x)