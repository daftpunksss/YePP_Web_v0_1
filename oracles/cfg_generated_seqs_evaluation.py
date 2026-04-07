import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

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
    
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List

tokenizer = AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision = "upstream_species_lm")

max_length = tokenizer.model_max_length

def kmers_stride1(seq: str, k: int = 6) -> List[str]:
    """将 DNA 序列按 1-bp 滑窗切成 k-mer 列表"""
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]

def tok_func_species(x, species_proxy, seq_col):
    res = tokenizer(species_proxy + " " +  " ".join(kmers_stride1(x[seq_col])))
    return res

def tok_func_standard(x, seq_col): return tokenizer(" ".join(kmers_stride1(x[seq_col])))

import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import os
from tqdm import tqdm
from datetime import datetime

class MultiSpeciesExpressionEvaluator:
    """
    使用四个SpeciesLM模型评估生成序列的表达水平
    """
    
    def __init__(self, 
             model_sc,
             model_pp,
             model_km,
             model_yl,
             model_km45,  # 新增
             model_io,    # 新增
             tokenizer,
             device: str = 'cuda'):
        self.model_sc = model_sc.to(device).eval()
        self.model_pp = model_pp.to(device).eval() 
        self.model_km = model_km.to(device).eval()
        self.model_yl = model_yl.to(device).eval()
        self.model_km45 = model_km45.to(device).eval()  # 新增
        self.model_io = model_io.to(device).eval()      # 新增
        self.tokenizer = tokenizer
        self.device = device

        self.species_proxies = {
            'sc': "kazachstania_africana_cbs_2517_gca_000304475",
            'pp': "komagataella_phaffii_gs115_gca_001746955", 
            'km': "kluyveromyces_marxianus_gca_001417835",
            'yl': "yarrowia_lipolytica_gca_014490615",
            'km45': "kluyveromyces_marxianus_gca_003335895",  # 示例
            'io': "pichia_kudriavzevii_gca_000764455"
        }

        self.models = {
            'sc': self.model_sc,
            'pp': self.model_pp,
            'km': self.model_km, 
            'yl': self.model_yl,
            'km45': self.model_km45,
            'io': self.model_io,
        }

        
    def sequence_to_kmers(self, seq: str, k: int = 6) -> List[str]:
        """将DNA序列转换为k-mers"""
        return [seq[i:i+k] for i in range(len(seq) - k + 1)]
    
    def evaluate_expression(self, 
                          sequence: str,
                          species_proxy: str,
                          model,
                          k: int = 6) -> float:
        """
        评估单个序列在特定物种中的表达水平
        """
        with torch.no_grad():
            text = species_proxy + " " + " ".join(self.sequence_to_kmers(sequence, k))
            tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            
            prediction = model(input_ids, attention_mask).item()
            
        return prediction
    
    def load_fasta_sequences(self, fasta_path: str) -> Tuple[List[str], List[str], List[str]]:
        """
        从FASTA文件加载序列，并从描述中提取物种名
        
        Returns:
            sequences: DNA序列列表
            headers: 完整的序列描述列表  
            species: 从描述中提取的物种名列表
        """
        sequences = []
        headers = []
        species = []
        
        with open(fasta_path, 'r') as f:
            current_seq = ""
            current_header = ""
            
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                        headers.append(current_header)
                        # 从描述中提取物种名（分割'_'后取最后一个）
                        species_name = current_header.split('_')[-1] if '_' in current_header else current_header
                        species.append(species_name)
                    current_header = line[1:]  # 去掉'>'符号
                    current_seq = ""
                else:
                    current_seq += line
            
            # 处理最后一个序列
            if current_seq:
                sequences.append(current_seq)
                headers.append(current_header)
                species_name = current_header.split('_')[-1] if '_' in current_header else current_header
                species.append(species_name)
        
        return sequences, headers, species
    
    def evaluate_all_sequences(self, 
                              sequences: List[str],
                              headers: List[str], 
                              species: List[str]) -> pd.DataFrame:
        """
        使用四个模型评估所有序列的表达水平
        """
        results = []
        
        print(f"Evaluating {len(sequences)} sequences with 4 models...")
        
        for i, (seq, header, spec) in enumerate(tqdm(zip(sequences, headers, species),
                                              desc="Evaluating sequences", 
                                              total=len(sequences))):
    
            pred_sc = self.evaluate_expression(seq, self.species_proxies['sc'], self.models['sc'])
            pred_pp = self.evaluate_expression(seq, self.species_proxies['pp'], self.models['pp'])
            pred_km = self.evaluate_expression(seq, self.species_proxies['km'], self.models['km'])
            pred_yl = self.evaluate_expression(seq, self.species_proxies['yl'], self.models['yl'])
            pred_km45 = self.evaluate_expression(seq, self.species_proxies['km45'], self.models['km45'])  # 新增
            pred_io = self.evaluate_expression(seq, self.species_proxies['io'], self.models['io'])        # 修复原来遗漏的变量定义

            results.append({
                'sequence_id': i + 1,
                'header': header,
                'species': spec,
                'sequence': seq,
                'sc_score': pred_sc,
                'pp_score': pred_pp, 
                'km_score': pred_km,
                'yl_score': pred_yl,
                'km45_score': pred_km45,  # 新增
                'io_score': pred_io       # 修复原来未定义
            })
        
        return pd.DataFrame(results)
    
    def save_results(self, df: pd.DataFrame, output_path: str = None) -> str:
        """
        将结果保存为CSV文件
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'expression_predictions_{timestamp}.csv'
        
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
        # 打印统计信息
        print(f"\nDataset Summary:")
        print(f"Total sequences: {len(df)}")
        print(f"Unique species: {df['species'].nunique()}")
        print(f"Species distribution:")
        print(df['species'].value_counts())
        
        print(f"\nExpression Score Statistics:")
        score_cols = ['sc_score', 'pp_score', 'km_score', 'yl_score', 'km45_score', 'io_score']
        print(df[score_cols].describe())
        
        return output_path

def load_models_and_tokenizer():
    """
    加载四个预训练模型和tokenizer
    注意：这里需要根据实际的模型架构和tokenizer进行调整
    """
    
    # 初始化模型
    model_sc = SpeciesLMLightAttention()
    model_pp = SpeciesLMLightAttention() 
    model_km = SpeciesLMLightAttention()
    model_yl = SpeciesLMLightAttention()
    model_km45 = SpeciesLMLightAttention()
    model_io = SpeciesLMLightAttention()
    
    # 加载预训练权重
    ckpt1 = torch.load('/root/autodl-tmp/graph_model/Diffusion_directed_evolution/SpeciesLM/finetune_ckpts/sc_ft_last2layers.ckpt')
    state_dict1 = {k.replace("model.", ""): v for k, v in ckpt1["state_dict"].items()}
    model_sc.load_state_dict(state_dict1, strict=False)
    
    ckpt2 = torch.load('/root/autodl-tmp/graph_model/Diffusion_directed_evolution/SpeciesLM/finetune_ckpts/pp_ft_last2layers.ckpt')
    state_dict2 = {k.replace("model.", ""): v for k, v in ckpt2["state_dict"].items()}
    model_pp.load_state_dict(state_dict2, strict=False)
    
    ckpt3 = torch.load('/root/autodl-tmp/graph_model/Diffusion_directed_evolution/SpeciesLM/finetune_ckpts/km_ft_last2layers.ckpt')
    state_dict3 = {k.replace("model.", ""): v for k, v in ckpt3["state_dict"].items()}
    model_km.load_state_dict(state_dict3, strict=False)
    
    ckpt4 = torch.load('/root/autodl-tmp/graph_model/Diffusion_directed_evolution/SpeciesLM/finetune_ckpts/yl_ft_last2layers.ckpt')
    state_dict4 = {k.replace("model.", ""): v for k, v in ckpt4["state_dict"].items()}
    model_yl.load_state_dict(state_dict4, strict=False)
    
    ckpt5 = torch.load('/root/autodl-tmp/graph_model/Diffusion_directed_evolution/SpeciesLM/finetune_ckpts/km45_ft_last2layers.ckpt')
    state_dict5 = {k.replace("model.", ""): v for k, v in ckpt5["state_dict"].items()}
    model_km45.load_state_dict(state_dict5, strict=False)
    
    ckpt6 = torch.load('/root/autodl-tmp/graph_model/Diffusion_directed_evolution/SpeciesLM/finetune_ckpts/io_ft_last2layers.ckpt')
    state_dict6 = {k.replace("model.", ""): v for k, v in ckpt6["state_dict"].items()}
    model_io.load_state_dict(state_dict6, strict=False)
    
    
    # 示例代码，需要根据实际情况修改
    print("Loading models and tokenizer...")
    
    return model_sc, model_pp, model_km, model_yl, model_km45, model_io

# 主函数
def main(fasta_path: str, output_path: str = None):
    """
    主函数：处理FASTA文件并生成预测结果
    
    Args:
        fasta_path: FASTA文件路径
        output_path: 输出CSV文件路径（可选）
    """
    
    # 加载模型和tokenizer
    model_sc, model_pp, model_km, model_yl, model_km45, model_io = load_models_and_tokenizer()

    evaluator = MultiSpeciesExpressionEvaluator(
        model_sc, model_pp, model_km, model_yl, model_km45, model_io, tokenizer
    )
    
    if model_sc is None:
        print("请先实现load_models_and_tokenizer函数来加载实际的模型")
        return
    
    # 加载FASTA序列
    print(f"Loading sequences from: {fasta_path}")
    sequences, headers, species = evaluator.load_fasta_sequences(fasta_path)
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"Species found: {set(species)}")
    
    # 评估所有序列
    results_df = evaluator.evaluate_all_sequences(sequences, headers, species)
    
    # 保存结果
    output_file = evaluator.save_results(results_df, output_path)
    
    return results_df, output_file

# 使用示例
if __name__ == "__main__":
    # 设置输入文件路径
    fasta_file = "/root/autodl-tmp/graph_model/Diffusion_directed_evolution/cfg_dfm_generated_results/dfm_generated_sequences_20251121_122433/all_generated_sequences.fasta"  # 替换为实际的FASTA文件路径
    output_file = "/root/autodl-tmp/graph_model/Diffusion_directed_evolution/SpeciesLM/AAA_plotting_code_and_file/chapeter3/strong_seed_scer_ppas_6_hosts.csv"  # 可选的输出文件路径
    
    # 运行预测
    try:
        results_df, output_path = main(fasta_file, output_file)
        print(f"\n预测完成！结果已保存到: {output_path}")
        
        # 显示前几行结果
        print(f"\n前5行结果预览:")
        print(results_df.head())
        
    except FileNotFoundError:
        print(f"错误：找不到FASTA文件 {fasta_file}")
        print("请确保文件路径正确")
    except Exception as e:
        print(f"运行时出错: {e}")
        print("请检查模型加载和依赖项")