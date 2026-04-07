import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class NT2B(nn.Module):
    def __init__(self, hidden_size=2560, dropout_prob=0.25):
        super(NT2B, self).__init__()
        #print(self.pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/graph_model/yeast_sequence_processing/DNABERT_and_other_pretrain_model/NT2.5b", trust_remote_code=True)
        self.pretrained_model = AutoModel.from_pretrained("/root/autodl-tmp/graph_model/yeast_sequence_processing/DNABERT_and_other_pretrain_model/NT2.5b", trust_remote_code=True)
        self.pretrained_model.resize_token_embeddings(len(self.tokenizer))
        for name, param in self.pretrained_model.named_parameters():
            if 'encoder.layer' in name:
                layer_number = int(name.split('.')[2])
                if layer_number < 28:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = False
                
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
    def forward(
        self,
        tokenized_seqs,
    ):
        
        attention_mask = tokenized_seqs != self.tokenizer.pad_token_id
        #print(f'tokenized_seqs.shape:{tokenized_seqs}')
        #print(f'attention_mask.shape:{attention_mask}')
        
        outputs = self.pretrained_model(
            input_ids=tokenized_seqs, 
            attention_mask=attention_mask,
            output_hidden_states=True)
        
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        
        normalized_output = self.layernorm(pooled_output)
        dropout_output = self.dropout(normalized_output)
        #print(f'dropout_output.shape:{dropout_output.shape}')
        fc1_output = torch.relu(self.fc1(dropout_output))
        logits = self.fc2(fc1_output)
        #print(f'logits.shape:{logits.shape}')
        return logits