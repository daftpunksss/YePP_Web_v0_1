import pandas as pd
import torch
import numpy as np
import h5py
from Bio import SeqIO
#from sklearn.preprocessing import QuantileTransformer
#from scipy.stats import boxcox

from torch.utils.data import Dataset
from transformers import AutoTokenizer

class OracleDataset(Dataset):
    """
    读取包含启动子序列及其 TP​M 标签的 CSV，并把序列
    编码成 LLM 可用的 token_id／attention_mask 张量。
    """
    def __init__(
        self,
        csv_path='/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/sc_constutive_oracle/deepexpression_dataset/sc_dataset_with_promoters_500_and_codon_frequency.csv',
        llm_model_path='/root/autodl-tmp/graph_model/yeast_sequence_processing/DNABERT_and_other_pretrain_model/NT100m',
        max_length: int | None = None,
    ):
        super().__init__()

        # ---------- 1. 载入数据 ----------
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["promoter_sequence", "TPM_avg_transformed"])  # 清理缺失值
        self.seqs = df["promoter_sequence"].astype(str).tolist()
        self.tpm_score = torch.tensor(
            df["TPM_avg_transformed"].values, dtype=torch.float
        )

        # ---------- 2. 初始化 tokenizer ----------
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_path, trust_remote_code=True
        )
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        # ---------- 3. 一次性批量 tokenize（更快） ----------
        encodings = self.tokenizer(
            self.seqs,
            padding="max_length",          # 固定长度，适合送进模型
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]          # [N, L]
        self.attention_mask = encodings["attention_mask"]

    # ---------- 4. Dataset 接口 ----------
    def __len__(self):
        return len(self.tpm_score)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.tpm_score[idx],   # Trainer 会自动取作为回归标签
        }


    
    

class OracleLLMDataset(torch.utils.data.Dataset):
    def __init__(self, mini_train=False, kldiv=False):
        self.shuffle = True
        self.token_ids_list = []
        self.tpm_score = []

        # self.tokenizer = AutoTokenizer.from_pretrained("./NT2.5b")
        # fasta_file = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/src/assigned_tpm_promoter700bp_upstream_of_gene.fasta'
        
        # for record in SeqIO.parse(fasta_file, 'fasta'):
        #     tokens_ids = self.tokenizer.batch_encode_plus([record.seq], return_tensors="pt", padding="max_length",
        #                                               max_length=self.tokenizer.model_max_length)["input_ids"]
        #     self.token_ids_list.append(tokens_ids)
        #     self.tpm_score.append(float(record.id))

        # self.tpm_score, _ = boxcox(np.array(self.tpm_score))
        #self.tpm_probs = np.load(kldiv_tpm)

        nt_embedding_file = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/src/nt_2.5B_assigned_with_annotations_500bp.h5'
        with h5py.File(nt_embedding_file, 'r') as llm:

            self.llm_emb = llm['embeddings'][:]
            self.tpm_score = llm['tpm'][:]
            self.tpm_score = np.array([float(x.decode('utf-8')) for x in self.tpm_score])
            self.tpm_score, _ = boxcox(self.tpm_score)
            
            # print(self.tpm_score.shape)
            # mask = self.tpm_score > 5
            # print(mask.shape)
            # filtered_tpm_score = np.log(self.tpm_score[mask])
            # print(filtered_tpm_score.shape)
            # filtered_llm_emb = self.llm_emb[mask]
            # print(filtered_llm_emb.shape)

            self.llm_emb = torch.tensor(self.llm_emb, dtype=torch.float32)
            self.tpm_score = torch.tensor(self.tpm_score, dtype=torch.float32)

            #筛选self.tpm_score>5的数据点，将对应的self.llm_emb和self.tpm_score读取为张量

        if not mini_train:
            #self.tpm_probs = torch.tensor(self.tpm_probs, dtype=torch.float32)
            self.tpm_score = self.tpm_score
        else:
            #self.token_ids_list = self.token_ids_list[:3000]
            self.llm_emb = self.llm_emb[:3000]
            #self.tpm_probs = torch.tensor(self.tpm_probs[:3000], dtype=torch.float32)
            self.tpm_score = self.tpm_score[:3000]

        
        #print(self.str_to_one_hot(self.seqs[0]))
    def __len__(self):
        return len(self.tpm_score)

    def __getitem__(self, idx):
        #seq = self.str_to_one_hot(self.seqs[idx]).clone().detach().float() # L, 4
        #probs = self.tpm_probs[idx].unsqueeze(-1)
        #llm_emb = self.llm_emb[idx]
        emb = self.llm_emb[idx]
        score = self.tpm_score[idx]
        return emb, score


from transformers import AutoTokenizer
from tqdm import tqdm
import re

class OracleTokenizedSeqDataset(torch.utils.data.Dataset):
    def __init__(self, mini_train=False):
        self.shuffle = True
        self.seqs = []
        self.tpm_score = []
        self.lw_tpm = []

        self.tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/graph_model/yeast_sequence_processing/DNABERT_and_other_pretrain_model/SegmentNT", trust_remote_code=True)
        fasta_file = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/src/cage/SceR/uas_promoter_expression_level.fasta'

        for record in SeqIO.parse(fasta_file, 'fasta'):
            pa = self.parse_seq_record(record)
            #if tpm is not None and tpm > 0:
            #if mrna is not None:
            #if pss > 10:
            self.seqs.append(str(record.seq).upper())
            self.tpm_score.append(pa)
            # self.lw_tpm.append(lw_tpm)
                #self.tpm_score.append(float(mrna))
            

        #self.tpm_score, _ = boxcox(np.array(self.tpm_score))
        # self.lw_tpm = np.log(np.array(self.lw_tpm)+1.0001)
        self.tpm_score = torch.tensor(self.tpm_score, dtype=torch.float32)
        # self.lw_tpm = torch.tensor(self.lw_tpm, dtype=torch.float32)
        #max_length = self.tokenizer.model_max_length
        max_length = 65

        all_input_ids = []
        batch_size = 1000
        for i in tqdm(range(0, len(self.seqs), batch_size), desc="Tokenizing sequences"):
            batch_seqs = self.seqs[i:i + batch_size]
            encoded_batch = self.tokenizer.batch_encode_plus(
                batch_seqs, 
                return_tensors="pt", 
                padding='max_length', #'max_length'
                #truncation=True,
                max_length=max_length, 
            )['input_ids']
            
            all_input_ids.append(encoded_batch)

        # 将所有批次的结果拼接成一个完整的张量
        
        self.tokenized_seq = torch.cat(all_input_ids, dim=0)
        print(f'shape of tokenized_seq:{self.tokenized_seq.shape}')
        # print('tokenizing sequences...')
        # self.tokenized_seq = self.tokenizer.batch_encode_plus(self.seqs, return_tensors="pt", padding=False, max_length=max_length, truncation=True)['input_ids']
        # print('seqence tokenized')
        #print(f'self.tokenized_seq len:{self.tokenized_seq}')
        #print(f"self.tokenized_seq.shape:{self.tokenized_seq.shape}")
        if not mini_train:
            self.tpm_score = self.tpm_score
            # self.lw_tpm = self.lw_tpm
        else:
            self.tokenized_seq = self.tokenized_seq[:300]
            self.tpm_score = self.tpm_score[:300]
            # self.lw_tpm = self.lw_tpm[:300]

    def __len__(self):
        return len(self.tpm_score)

    def __getitem__(self, idx):
        
        tokenized_seqs = self.tokenized_seq[idx]
        score = self.tpm_score[idx]
        tpm_score = self.tpm_score[idx]

        return tokenized_seqs, tpm_score
    
    def parse_seq_record(self, record):
        # 从description中提取tpm和ctss_count
        # parts = record.description.replace(" ", "").split(',')

        # tpm = float(parts[0].split('=')[1])
        # mrna = float(parts[1].split('=')[1])
        # pss = float(parts[2].split('=')[1])
        
        # if mrna != 'NA':
        #     mrna = float(parts[1].split('=')[1])
        #     return tpm, mrna, pss
        # else:
        #     return tpm, None, pss
        parts = record.description.replace(" ", "").split(',')
        pa = np.log(float(parts[1])+1.00001)
        #pa = float(parts[1])
        return pa
    
    
class OracleLegNetDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.shuffle = True
        self.seqs = []
        self.pa = []
        self.max_len = 1000
        fasta_file = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/pichia_rnaseq/filtered_promoter_sequences.fasta'
        
        for record in SeqIO.parse(fasta_file, 'fasta'):
            seq_str = str(record.seq).upper()
            truncated_seq = self.truncate_sequence(seq_str)
            self.seqs.append(truncated_seq)
            self.pa.append(self.parse_seq_record(record))
        
        print(self.pa[:20])

    def __len__(self):
        return len(self.pa)

    def __getitem__(self, idx):
        seq = self.str_to_one_hot(self.seqs[idx]).clone().detach().float() # L, 4
        score = torch.tensor(self.pa[idx],dtype=torch.float)
        return seq, score
    
    def str_to_one_hot(self, seq):
        base_map = {'A':0, 'C':1, 'G':2, 'T':3}
        seq_tensor = torch.tensor([base_map[base] for base in seq], dtype=torch.long)
        seq_one_hot = torch.nn.functional.one_hot(seq_tensor, num_classes=4)
        return seq_one_hot
    
    def truncate_sequence(self, seq):
        if len(seq) > self.max_len:
            return seq[-self.max_len:]
        return seq
    
    def parse_seq_record(self, record):
        parts = record.description.strip(' ').split(',')
        #pa = np.log(float(parts[1])+1.00001)
        pa = float(parts[1])
        return pa
    
import torch
from Bio import SeqIO
import pandas as pd

class OracleHybridNetDataset(torch.utils.data.Dataset):
    def __init__(self, max_len=1000):
        self.seqs = []
        self.codon_freqs = []
        self.total_codons = []
        self.motif_count = []
        self.pa = []
        self.max_len = max_len
        self.embeddings = []
        codon_freq_file = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/pichia_rnaseq/codon_frequencies.csv'
        fasta_file = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/pichia_rnaseq/filtered_promoter_sequences.fasta'
        embedding_file = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/pichia_rnaseq/nt_2.5B_pichia_1000bp.h5'
        #motif_count_file = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/pichia_rnaseq/motif_counts.csv'
        # 读取密码子频数数据
        codon_freq_data = pd.read_csv(codon_freq_file,sep=',')
        print(codon_freq_data.head())
        self.codon_freq_dict = {
            row['gene_name']: row for _, row in codon_freq_data.iterrows()
        }
        
        # motif_count = pd.read_csv(motif_count_file)
        # self.motif_count_dict = {
        #     row['gene_name']: row for _, row in motif_count.iterrows()
        # }

        # 首先读取所有序列的embeddings
        
        # 读取 Fasta 文件并匹配密码子频数
        with h5py.File(embedding_file, 'r') as h5f:
            embeds = h5f['embeddings'][:]
            print(embeds.shape)
            for idx, record in enumerate(SeqIO.parse(fasta_file, 'fasta')):
                seq_str = str(record.seq).upper()
                truncated_seq = self.truncate_sequence(seq_str)
                self.seqs.append(truncated_seq)
                
                gene_name, pa = self.parse_seq_record(record)
                self.pa.append(pa)
                if gene_name in self.codon_freq_dict:
                    #在这里使用enumrate()获取对应的idx, 然后使用idx提取embeddings，同时处理成（480，4）的形状
                    embedding = torch.tensor(embeds[idx], dtype=torch.float)  # 转换为张量
                    reshaped_embedding = embedding.view(640, 4)  # 转换形状
                    self.embeddings.append(reshaped_embedding)
                    codon_freq_row = self.codon_freq_dict[gene_name]
                    codon_freq_array = codon_freq_row[1:65].astype(float).values.reshape(16, 4)
                    # motif_count_row = self.motif_count_dict[gene_name]
                    # motif_count_array = motif_count_row[1:179].astype(float).values
                    # motif_count_array = np.pad(
                    #                 motif_count_array,
                    #                 pad_width=(0, 2),  # 在末尾填充
                    #                 mode='constant',
                    #                 constant_values=0
                    #             ).reshape(45, 4)
                    # self.motif_count.append(motif_count_array)
                    self.codon_freqs.append(codon_freq_array)  # 64个密码子频数转为(16, 4)
                    total_codon_feature = [codon_freq_row['total_codons']] * 4
                    self.total_codons.append(total_codon_feature)  # 转为(1, 4)格式
                else:
                    print(f"Warning: Gene {gene_name} not found in codon frequency file.")
                    embedding = torch.tensor(embeds[idx], dtype=torch.float)  # 转换为张量
                    reshaped_embedding = embedding.view(640, 4)  # 转换形状
                    self.embeddings.append(reshaped_embedding)  
                    self.codon_freqs.append(torch.zeros(16, 4))
                    self.total_codons.append([0] * 4)
                    # motif_count_row = self.motif_count_dict[gene_name]
                    # motif_count_array = motif_count_row[1:179].astype(float).values
                    # motif_count_array = np.pad(
                    #                 motif_count_array,
                    #                 pad_width=(0, 2),  # 在末尾填充
                    #                 mode='constant',
                    #                 constant_values=0
                    #             ).reshape(45, 4)
                    # self.motif_count.append(motif_count_array)
        
        # 在这里把选取到的embeddings使用torch.stack转换成整块的张量
        self.embeddings = torch.stack(self.embeddings)  # (N, 480, 4)
        self.seqs = torch.stack([self.str_to_one_hot(seq).clone().detach().float() for seq in self.seqs])
        self.codon_freqs = torch.stack([torch.tensor(freq, dtype=torch.float) for freq in self.codon_freqs])  # (N, 16, 4)
        self.total_codons = torch.stack([torch.tensor(total, dtype=torch.float) for total in self.total_codons]).unsqueeze(1)  # (N, 1, 4)
        #self.motif_counts = torch.stack([torch.tensor(mc, dtype=torch.float) for mc in self.motif_count])
        print(f"Warning: Gene {len(codon_freq_data)-len(gene_name)} not found in codon frequency file.")  
        print(len(self.seqs))
        print(len(self.codon_freqs))
        print(len(self.total_codons))
        #print(self.motif_counts.shape)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        # One-hot 序列特征 (L, 4)
        seq = self.seqs[idx]
        freq = self.codon_freqs[idx]
        length = self.total_codons[idx]
        embedding = self.embeddings[idx]
        #motif_count = self.motif_counts[idx]
        score = torch.tensor(self.pa[idx], dtype=torch.float)
        
        # 密码子频数特征 (16, 4)
        # codon_freq_tensor = torch.tensor(self.codon_freqs[idx]).clone().detach().float()  # (16, 4)

        # 密码子总数特征 (1, 4)
        # total_codon_tensor = torch.tensor(self.total_codons[idx], dtype=torch.float).view(1, 4)  # (1, 4)
        # print(seq.shape)
        # print(self.codon_freqs.shape)
        # print(self.total_codons.shape)
        # 拼接所有特征
        combined_features = torch.cat((seq, freq, length, embedding), dim=0)  # (L + 16 + 1, 4)
        # print(combined_features.shape)
        # print(score)
        return combined_features, score  # 返回拼接后的特征和表达量

    def str_to_one_hot(self, seq):
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        seq_tensor = torch.tensor([base_map[base] for base in seq], dtype=torch.long)
        seq_one_hot = torch.nn.functional.one_hot(seq_tensor, num_classes=4)
        return seq_one_hot

    def truncate_sequence(self, seq):
        if len(seq) > self.max_len:
            return seq[-self.max_len:]
        return seq

    def parse_seq_record(self, record):
        parts = record.description.split(',')
        #pa = np.log(float(parts[1])+1.00001)
        pa = float(parts[1])
        name = str(parts[0]).strip(' ')
        #print(name)
        return name,pa

import torch
import pandas as pd
import numpy as np

class SCOracleHybridNetDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.seqs = []
        self.pa = []
        cnsp = '/root/autodl-tmp/graph_model/SWAT/promoterxpositioneffect/sc_prom_with_intfeat.csv'
        emb = '/root/autodl-tmp/graph_model/SWAT/promoterxpositioneffect/sc_prom_final_3832.h5'
        cnsp_data = pd.read_csv(cnsp)
        
        codon_frequencies = cnsp_data.iloc[:, -65:-1].values  # 取64个密码子的频率
        codon_count = cnsp_data.iloc[:, -10:].values  # 取密码子总数
        print(codon_count.shape)
        reshaped_frequencies = codon_frequencies.reshape(-1, 16, 4)
        expanded_codon_count = np.expand_dims(codon_count, axis=-1)  # shape: (n_samples, 1)
        expanded_codon_count = np.tile(expanded_codon_count, (1,1,4))  # shape: (n_samples, 4)
        #expanded_codon_count = np.expand_dims(expanded_codon_count, axis=1)  # shape: (n_samples, 4, 1)
        
        print(expanded_codon_count.shape)
        print(reshaped_frequencies.shape)
        self.codon_data = np.concatenate([reshaped_frequencies, expanded_codon_count], axis=1)  # shape: (n_samples, 4, 17)
        
        self.seqs = cnsp_data['promoter_sequence'].tolist()
        #print(self.seqs)
        #self.pa = np.log1p(cnsp_data['TPM_avg_transformed'])
        self.pa = cnsp_data['TPM_avg_transformed'].tolist()

        self.seqs = [self.str_to_one_hot(seq) for seq in self.seqs]
        self.seqs = torch.stack(self.seqs)  # Stack sequences into a single tensor
        #print(self.seqs.shape)
        self.seqs = torch.cat([self.seqs, torch.tensor(self.codon_data, dtype=torch.float32)], dim=1)  # 拼接
        #print(self.seqs.shape)
        #将self.seqs与emb进行拼接
        with h5py.File(emb, 'r') as f:
            seq_emb = f['embeddings'][:]
            print(seq_emb.shape)
            seq_emb = torch.tensor(seq_emb, dtype=torch.float32).reshape(-1, 768, 4) #genera (-1, 768, 4) NT2.5B (-1, 640, 4)

        self.seqs = torch.cat([self.seqs, seq_emb], dim=1)
        #print(self.seqs.shape)
        self.pa = torch.tensor(self.pa, dtype=torch.float32)

    def __len__(self):
        return len(self.pa)

    def __getitem__(self, idx):
        return self.seqs[idx], self.pa[idx]

    def str_to_one_hot(self, seq):
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        #print(f'sequence length: {len(seq)}')
        if pd.isna(seq) or len(seq) < 500:
            print(f'using random promoter sequences...')
            seq_length = len(seq) if not pd.isna(seq) else 0  # 处理 NaN 值，设定长度为 0
            padding_length = 500 - seq_length
            padding = torch.full((padding_length,), 2, dtype=torch.long)  # 填充为 "G" (即2)
            seq_tensor = torch.tensor([base_map[base] for base in seq if base in base_map], dtype=torch.long)
            if not pd.isna(seq):
                seq_tensor = torch.cat([seq_tensor, padding])
            else:
                seq_tensor = padding  # 如果是 NaN，则直接用填充代
        else:
            seq_tensor = torch.tensor([base_map[base] for base in seq if base in base_map], dtype=torch.long)
        seq_one_hot = torch.nn.functional.one_hot(seq_tensor, num_classes=4).float()
        return seq_one_hot



import torch
import pandas as pd
import numpy as np

class CrossSpeOracleHybridNetDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.seqs = []
        self.pa = []
        cnsp = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/sc_constutive_oracle/deepexpression_dataset/sc_dataset_with_promoters_and_codon_frequency.csv'
        g_emb = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/sc_constutive_oracle/deepexpression_dataset/sc_gene_genera_embeddings.h5'
        p_emb = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/sc_constutive_oracle/deepexpression_dataset/sc_promoter_genera_embeddings.h5'
        cnsp_data = pd.read_csv(cnsp)
        
        codon_frequencies = cnsp_data.iloc[:, -65:-1].values  # 取64个密码子的频率
        codon_count = cnsp_data.iloc[:, -1].values  # 取密码子总数
        
        reshaped_frequencies = codon_frequencies.reshape(-1, 16, 4)
        expanded_codon_count = np.expand_dims(codon_count, axis=1)  # shape: (n_samples, 1)
        expanded_codon_count = np.tile(expanded_codon_count, (1,4))  # shape: (n_samples, 4)
        expanded_codon_count = np.expand_dims(expanded_codon_count, axis=1)  # shape: (n_samples, 4, 1)
        
        print(expanded_codon_count.shape)
        print(reshaped_frequencies.shape)
        self.codon_data = np.concatenate([reshaped_frequencies, expanded_codon_count], axis=1)  # shape: (n_samples, 4, 17)
        
        #self.seqs = cnsp_data['promoter_sequence'].tolist()
        #self.pa = np.log1p(cnsp_data['TPM_avg_transformed'])
        self.pa = cnsp_data['TPM_avg_transformed'].tolist()

        self.seqs = torch.tensor(self.codon_data, dtype=torch.float32)  # 拼接
        print(self.seqs.shape)
        # 将self.seqs与emb进行拼接
        with h5py.File(g_emb, 'r') as f:
            g_emb = f['embeddings'][:]
            print(g_emb.shape)
            g_emb = torch.tensor(g_emb, dtype=torch.float32).reshape(-1,768, 4) #genera (-1, 768, 4) NT2.5B (-1, 640, 4)
            
        with h5py.File(p_emb, 'r') as f:
            p_emb = f['embeddings'][:]
            print(p_emb.shape)
            p_emb = torch.tensor(p_emb, dtype=torch.float32).reshape(-1, 768, 4) #genera (-1, 768, 4) NT2.5B (-1, 640, 4)

        self.seqs = torch.cat([p_emb, self.seqs, g_emb], dim=1)
        #print(self.seqs.shape)
        self.pa = torch.tensor(self.pa, dtype=torch.float32)

    def __len__(self):
        return len(self.pa)

    def __getitem__(self, idx):
        return self.seqs[idx], self.pa[idx]

    def str_to_one_hot(self, seq):
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        seq_tensor = torch.tensor([base_map[base] for base in seq if base in base_map], dtype=torch.long)
        if len(seq_tensor) < 500:
            padding_length = 500 - len(seq_tensor)
            padding = torch.full((padding_length,), 2, dtype=torch.long)  # 填充为 "G" (即2)
            seq_tensor = torch.cat([seq_tensor, padding])
        seq_one_hot = torch.nn.functional.one_hot(seq_tensor, num_classes=4).float()
        return seq_one_hot
    
    
import torch
import pandas as pd
import h5py
import numpy as np
from torch.utils.data import Dataset
from scipy.stats import boxcox

class CssHybridNetDataset(Dataset):
    def __init__(self, max_len=1000):
        self.seqs = []
        self.codon_freqs = []
        self.total_codons = []
        self.embeddings = []
        self.pa = []
        self.max_len = max_len

        # 文件路径
        codon_freq_file = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/12_cage_oracle/codon_frequencies/ waltii_codon_frequencies.csv'
        seq_file = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/12_cage_oracle/sequences/ waltii_promoter_sequences.csv'
        embedding_file = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/12_cage_oracle/embeddings/ waltii_embeddings.h5'

        # 读取密码子频率文件
        codon_freq_data = pd.read_csv(codon_freq_file)
        self.codon_freq_dict = {
            row['gene_name']: row for _, row in codon_freq_data.iterrows()
        }

        # 读取序列文件
        seq_data = pd.read_csv(seq_file)
        # 将seq整理为gene_name: seq的字典，名为self.seq_dict
        self.seq_dict = {row['gene_name']: row['promoter_seq'] for _, row in seq_data.iterrows()}
        pa, _ = boxcox(np.array(seq_data['TPM*']+ 1e-5))
        gene_ids = seq_data['gene_name'].tolist()

        # 读取嵌入文件
        with h5py.File(embedding_file, 'r') as h5f:
            embeds = h5f['embeddings'][:]
            

            for idx, gene_name in enumerate(gene_ids):
                
                #seq = self.seq_dict[gene_name]
                #if seq<1000:
                #   continue（跳过当前的idx)
                seq = self.seq_dict.get(gene_name, None)
                
                if seq is None or len(seq) < 1000:
                    continue
                else:
                    self.seqs.append(seq)
                    codon_freq_row = self.codon_freq_dict[gene_name]
                    codon_freq_array = codon_freq_row[1:65].astype(float).values.reshape(16, 4)
                    self.codon_freqs.append(codon_freq_array)  # 64个密码子频数转为(16, 4)
                    total_codon_feature = [codon_freq_row['total_codons']] * 4
                    self.total_codons.append(total_codon_feature)  # 转为(1, 4)格式
                    embedding = torch.tensor(embeds[idx], dtype=torch.float).view(640, 4)
                    self.embeddings.append(embedding)
                    self.pa.append(pa[idx])

        # 转换为张量
        self.embeddings = torch.stack(self.embeddings)  # (N, 480, 4)
        self.seqs = torch.stack([self.str_to_one_hot(seq).clone().detach().float() for seq in self.seqs])
        self.codon_freqs = torch.stack([torch.tensor(freq, dtype=torch.float) for freq in self.codon_freqs])  # (N, 16, 4)
        self.total_codons = torch.stack([torch.tensor(total, dtype=torch.float) for total in self.total_codons]).unsqueeze(1)
        
        
        print(f'self.embeddings shape: {self.embeddings.shape}')
        print(f'self.seqs shape: {self.seqs.shape}')
        print(f'self.codon_freqs shape: {self.codon_freqs.shape}')
        print(f'self.total_codons shape: {self.total_codons.shape}')
        print(f'self.pa shape: {len(self.pa)}')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        freq = self.codon_freqs[idx]
        length = self.total_codons[idx]
        embedding = self.embeddings[idx]
        combined_features = torch.cat((seq, freq, length, embedding), dim=0)
        #print(combined_features.shape)
        score = torch.tensor(self.pa[idx], dtype=torch.float)
    
        return combined_features, score

    def str_to_one_hot(self, seq):
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        seq_tensor = torch.tensor([base_map[base] for base in seq], dtype=torch.long)
        seq_one_hot = torch.nn.functional.one_hot(seq_tensor, num_classes=4)
        return seq_one_hot.float()

    def truncate_sequence(self, seq):
        if len(seq) > self.max_len:
            return seq[-self.max_len:]
        return seq

import torch
import pandas as pd
import numpy as np
import h5py
from scipy.stats import boxcox

class TranslationLeadersDataset(torch.utils.data.Dataset):
    def __init__(self):
        #te_file_path = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/Kozak_20250528/lk_for_fluo_pred.csv'
        #emb_file_path = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/Kozak_20250528/tleaders_10203.h5'
        #kozak_csv = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/Kozak_20250528/kozak_score.csv'
        
        # with h5py.File(emb_file_path, 'r') as h5f:
        #     self.embeds = h5f['embeddings'][:]
        
        #te = pd.read_csv(te_file_path,sep='\t')
        
        te = pd.read_csv('/root/autodl-tmp/graph_model/Diffusion_directed_evolution/Kozak_20250528/features_engineering/lk_info_with_features.csv')
        # self.tleads = [k + u[-4:] for k, u in zip(te['UTRseq'], te['Kozak'])]
        
        non_feat  = ["gene_id", "lk_fluo", "prom_seqs"]
        feat_df  = te.drop(columns=non_feat)
        self.feature_cols = feat_df.select_dtypes(include=[np.number]).columns
        self.features = feat_df[self.feature_cols].to_numpy(dtype=np.float32)
        
        # columns_to_avg = ['Glucose', 'Fructose', 'Sorbitol 1M', 'Growth at 39o', 'NaCl 1M',
        #           'Glucose lacking amino acids', 'Galactose lacking amino acids',
        #           'Galactose', 'Glycerol', 'Ethanol']
        self.te = te['lk_fluo']
        # self.te,_= boxcox(np.array(te['MeanYFP'])) # , lmbda=0.25
        
        print(len(self.te))
        print(te)
        
        # kz_df            = pd.read_csv(kozak_csv,sep='\t')                    # cols: 8mer, score
        # self.kz_dict     = {row['8mer'].upper(): row['score'] for _, row in kz_df.iterrows()}
        # self.default_kz  = 0.0  
        
        ## other features 
        ## 'Length', 'U7', 'uAUG', 'FreqA', 'FreqT', 'FreqG', 'FreqC','DDG_Median', 'DDG_AVG', 'Gquads', 'Cap40ntFodling',
        #    'Cap40ntFodling_Absolute_Value', 'LSMKozakStart', 'UMax',
        #    'Cap_Proximal_A', 'Cap_Proximal_C', 'Cap_Proximal_G', 'Cap_Proximal_T',
        #    'Distal_A', 'Distal_C', 'Distal_G', 'Distal_T',
        # self.utr_len = te['Length']
        # self.uAUG = te['uAUG']
        # self.freqA = te['FreqA']
        # self.freqG = te['FreqG']
        # self.freqC = te['FreqC']
        # self.freqT = te['FreqT']
        # # self.Gquads = te['Gquads']
        # # self.ddg = te['DDG_Median']
        # self.Cap40ntFodling = te['Cap40ntFodling']
        # self.LSMKozakStart = te['LSMKozakStart']
        # self.Cap_Proximal_A = te['Cap_Proximal_A']
        # self.Cap_Proximal_C = te['Cap_Proximal_C']
        # self.Cap_Proximal_G = te['Cap_Proximal_G']
        # self.Cap_Proximal_T = te['Cap_Proximal_T']
        # self.Distal_A = te['Distal_A']
        # self.Distal_C = te['Distal_C']
        # self.Distal_G = te['Distal_G']
        # self.Distal_T = te['Distal_T']
        # self.sc_pred = te['sc_pred']
        # self.MeanRNA = te['MeanRNA']
        

    def __len__(self):
        return len(self.te)
    
    @staticmethod
    def count_ATG(seq: str) -> int:
        """可重叠统计 ATG 次数"""
        seq = seq.upper()
        return sum(1 for i in range(len(seq) - 2) if seq[i:i+3] == 'ATG')
    
    def calc_lsm_score(self, seq: str) -> float:
        """按 Leaky-Scanning Model 计算 adjusted Kozak score (P_CDS)"""
        seq = seq.upper()
        aug_pos = [i for i in range(len(seq) - 2) if seq[i:i+3] == 'ATG']
        if not aug_pos:
            return 0.0

        p_list = []
        for pos in aug_pos:
            upstream = seq[max(0, pos-4):pos]
            upstream = upstream.rjust(4, 'N')
            plus1    = seq[pos+3] if pos+3 < len(seq) else 'N'
            kmer8    = upstream + 'ATG' + plus1
            p_list.append(self.kz_dict.get(kmer8, self.default_kz))

        p_main = p_list[-1]
        p_skip = 1.0
        for p in p_list[:-1]:
            p_skip *= (1.0 - p)

        return p_main * p_skip

    def __getitem__(self, idx):

        # one_hot_seq = self.str_to_one_hot(self.tleads[idx], l=100).T
        # emb = torch.tensor(self.embeds[idx], dtype=torch.float).reshape(-1,4,768).squeeze(0)
        
        # kz_val  = self.calc_lsm_score(self.tleads[idx])
        # atg_val = self.count_ATG(self.tleads[idx])
        # kz_score  = torch.full((4, 1), kz_val,  dtype=torch.float)        # (4,1)
        # atg_count = torch.full((4, 1), atg_val, dtype=torch.float)        # (4,1)
        # utr_len = torch.full((4, 1), self.utr_len[idx],  dtype=torch.float)
        # uAUG = torch.full((4, 1), self.uAUG[idx],  dtype=torch.float)
        # Gquads = torch.full((4, 1), self.Gquads[idx],  dtype=torch.float)
        # ddg = torch.full((4, 1), self.ddg[idx],  dtype=torch.float)
        # LSMKozakStart = torch.full((4, 1), self.LSMKozakStart[idx],  dtype=torch.float)
        # freqA = torch.full((4, 1), self.freqA[idx],  dtype=torch.float)
        # freqG = torch.full((4, 1), self.freqG[idx],  dtype=torch.float)
        # freqC = torch.full((4, 1), self.freqC[idx],  dtype=torch.float)
        # freqT = torch.full((4, 1), self.freqT[idx],  dtype=torch.float)
        # Cap40ntFodling = torch.full((4, 1), self.Cap40ntFodling[idx],  dtype=torch.float)
        # Cap_Proximal_A = torch.full((4, 1), self.Cap_Proximal_A[idx],  dtype=torch.float)
        # Cap_Proximal_C = torch.full((4, 1), self.Cap_Proximal_C[idx],  dtype=torch.float)
        # Cap_Proximal_T = torch.full((4, 1), self.Cap_Proximal_T[idx],  dtype=torch.float)
        # Cap_Proximal_G = torch.full((4, 1), self.Cap_Proximal_G[idx],  dtype=torch.float)
        # Distal_A = torch.full((4, 1), self.Distal_A[idx],  dtype=torch.float)
        # Distal_T = torch.full((4, 1), self.Distal_T[idx],  dtype=torch.float)
        # Distal_C = torch.full((4, 1), self.Distal_C[idx],  dtype=torch.float)
        # Distal_G = torch.full((4, 1), self.Distal_G[idx],  dtype=torch.float)
        # sc_pred = torch.full((4, 1), self.sc_pred[idx],  dtype=torch.float)
        #MeanRNA = torch.full((4, 1), self.MeanRNA[idx],  dtype=torch.float)
        
        # combined = torch.cat([
        #                     #  one_hot_seq, 
        #                       #utr_len,
        #                       #Gquads, 
        #                       #ddg, 
        #                       uAUG,
        #                       LSMKozakStart, 
        #                       freqA, freqG, freqC, freqT,
        #                       Cap40ntFodling, 
        #                       Cap_Proximal_A, Cap_Proximal_C, Cap_Proximal_T, Cap_Proximal_G,
        #                       Distal_A, Distal_T, Distal_C, Distal_G,
        #                       sc_pred,
        #                       #MeanRNA, 
        #                       #emb
        #                       ], dim=1)        
        
        features =  torch.tensor(self.features[idx], dtype=torch.float)  
        score = torch.tensor(self.te[idx], dtype=torch.float)
        
        return features, score

    def str_to_one_hot(self, seq, l=50):
        """Right-truncate to last 50bp, then zero-padding and one-hot encoding."""
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        # Encode sequence
        encoded = [base_map.get(base, -1) for base in seq]
        encoded = [i for i in encoded if i >= 0]

        # 取最后 l 个碱基，不足则padding
        if len(encoded) >= l:
            encoded = encoded[-l:]
            padded = encoded
        else:
            pad_len = l - len(encoded)
            padded = [-1] * pad_len + encoded  # 在左侧 padding

        # One-hot 编码
        out = torch.zeros((l, 4), dtype=torch.float)
        for i, idx in enumerate(padded):
            if idx in [0, 1, 2, 3]:
                out[i, idx] = 1.0
        return out

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re
from scipy.stats import boxcox

class YeastTSSProphetDataset(Dataset):
    """
    每条样本输出:
        seq_onehot : [L, 4]           (L = 504)
        reg_labels : [3] or [1]       (TPM 已 BoxCox、UTR_len、PSS)
        p_target   : [L]              (TSS 概率分布, sum=1)
    """

    def __init__(self,
                 tsv_path="/root/autodl-tmp/graph_model/Diffusion_directed_evolution/YeastTSSProphet/data/tss_train.tsv",
                 seq_len: int = 504):
        super().__init__()
        self.seq_len = seq_len

        # ---------- 1. 读取 & 清洗 ----------
        df = (pd.read_csv(tsv_path, sep="\t")
                .rename(columns=lambda c: c.strip())
                .dropna(subset=["DNA_seq", "TPM*", "Weighted 5'UTR length*", "PSS*"])
                .assign(DNA_seq=lambda d: d["DNA_seq"].str.upper())
        )

        # ---------- 2. Box-Cox 变换 TPM ----------
        df["TPM"] = boxcox(df["TPM*"] + 1e-3, lmbda=0.2)   # 避免 0

        # ---------- 3. 预生成张量 ----------
        self.seqs   = [self.str_to_one_hot(s) for s in df["DNA_seq"]]

        self.labels = torch.tensor(
            df[["TPM", "Weighted 5'UTR length*", "PSS*"]].values,
            dtype=torch.float32
        )

        # ---------- 4. 构造目标分布 ----------
        self.p_targets = [
            self.build_target_distribution(
                utr_len=row["Weighted 5'UTR length*"],
                pss=row["PSS*"],
                tpm=row["TPM"]        # 若只做形状，可不传 tpm
            )
            for _, row in df.iterrows()
        ]

    # -------- Dataset 接口 --------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        返回:
            seq_onehot [4, L]   (已转为 Conv1D 习惯 shape)
            reg_labels [3]
            p_target   [L]
        """
        seq = self.seqs[idx].T               # [L,4] -> [4,L]
        return seq, self.p_targets[idx]

    # -------- 帮助函数 --------
    def str_to_one_hot(self, seq: str) -> torch.FloatTensor:
        """把 DNA 序列转 1-hot，固定长度 seq_len；非法字符全 0。"""
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        seq = re.sub(r"[^ACGT]", "", seq)
        seq = seq[-self.seq_len:] if len(seq) >= self.seq_len else seq + "N" * (self.seq_len - len(seq))

        idxs = [base_map.get(b, -1) for b in seq]
        one_hot = torch.zeros(self.seq_len, 4, dtype=torch.float32)
        for pos, idx in enumerate(idxs):
            if idx >= 0:
                one_hot[pos, idx] = 1.0
        return one_hot

    # -------- 目标分布构造 --------
    def build_target_distribution(self,
                                  utr_len: float,
                                  pss: float,
                                  tpm: float,
                                  min_sigma: float = 1.0) -> torch.FloatTensor:
        """
        根据 5′UTR_len / PSS / TPM 构造长度 = seq_len 的高斯概率分布。
        - μ : 触发位点坐标，ATG 设在索引 seq_len-1
        - σ : 用 PSS/4 近似；>= min_sigma 保证非 0
        - A : 总量 (tpm)，仅作归一化前振幅，可留作权重
        返回归一化后 (sum = 1) 的 tensor
        """
        μ = self.seq_len - int(round(utr_len)) - 1          # 0-based index
        σ = max(pss / 4.0, min_sigma)

        x = torch.arange(self.seq_len, dtype=torch.float32)
        dist = torch.exp(-0.5 * ((x - μ) / σ) ** 2)

        # 振幅可乘 tpm，但做 KL 前需归一化；这里直接归一化即可
        p = dist / dist.sum()

        return p


import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re

class YeastTSSProphetDataset2(Dataset):
    """
    每条样本输出:
        seq_onehot : [L, 4]  (L = 504)
        cage_signal: [1, 500]
    """

    # ---------- 通用工具 ----------
    @staticmethod
    def _soft_clip(x, tc: float = 384.0):
        """f(x) = min(x, tc + √max(0, x − tc)) ；支持向量化"""
        x = np.asarray(x, dtype=np.float32)
        return np.minimum(x, tc + np.sqrt(np.maximum(0.0, x - tc)))

    def __init__(self,
                 tsv_path: str = "/root/autodl-tmp/graph_model/Diffusion_directed_evolution/YeastTSSProphet/Dataset13_for_train.tsv",
                 seq_len: int = 504):
        super().__init__()
        self.seq_len = seq_len
        self.window  = 500               # CAGE 信号长度

        # ---------- 1. 读取 & 清洗 ----------
        df = (
            pd.read_csv(tsv_path, sep="\t")
              .rename(columns=lambda c: c.strip())
              .dropna(subset=[
                  "TC Width", "Dominant TSS site", "TPM of TC",
                  "TPM of dominant TSS", "utr_len",
                  "TC_start_5'_start_as_base1", "TC_end_5'_start_as_base1",
                  "DNA_seq"
              ])
              .assign(DNA_seq=lambda d: d["DNA_seq"].str.upper())
        )

        # ---------- 2. soft-clip 规范 TPM 值 ----------
        df["TC_TPM_clip"]   = self._soft_clip(df["TPM of TC"])
        df["Dom_TPM_clip"]  = self._soft_clip(df["TPM of dominant TSS"])

        # ---------- 3. 预生成序列 one-hot ----------
        self.seqs = [self._str_to_one_hot(s) for s in df["DNA_seq"]]

        # ---------- 4. 构造 CAGE-signal 张量 ----------
        cage_signals = []
        #cage_values = []
        for _, row in df.iterrows():

            signal = np.zeros(self.window, dtype=np.float32)

            # 4-1  dominant 位置（0-based）
            apex_idx = int(self.window - row["utr_len"])
            if 0 <= apex_idx < self.window:
                signal[apex_idx] = row["Dom_TPM_clip"]

            # 4-2  其余 TC 区间均匀分配
            rest = max(row["TC_TPM_clip"] - row["Dom_TPM_clip"], 0.0)
            start_idx = int(row["TC_start_5'_start_as_base1"]) - 1
            end_idx   = int(row["TC_end_5'_start_as_base1"])   - 1
            width = end_idx - start_idx + 1

            if rest > 0 and width > 1:
                per_base = rest / (width - 1)
                for pos in range(start_idx, end_idx + 1):
                    if pos == apex_idx or not (0 <= pos < self.window):
                        continue
                    signal[pos] = per_base
            # width==0 或 rest==0 → 仅有 dominant ，已在 4-1 赋值

            cage_signals.append(torch.tensor(signal, dtype=torch.float32))
            #cage_values.append(torch.tensor(df["TC_TPM_clip"], dtype=torch.float32))

        self.cage_signals = cage_signals
        self.cage_values = torch.tensor(np.array(df["TC_TPM_clip"]), dtype=torch.float32)

    # -------- Dataset 接口 --------
    def __len__(self):
        return len(self.cage_signals)

    def __getitem__(self, idx):
        """
        返回:
            seq_onehot : [L, 4]
            cage_signal: [1, 500]
        """
        seq = self.seqs[idx]            # [L,4]
        return seq, self.cage_signals[idx], self.cage_values[idx]

    # -------- 帮助函数 --------
    def _str_to_one_hot(self, seq: str) -> torch.FloatTensor:
        """DNA → one-hot；固定长度 seq_len；非法字符全 0。"""
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        seq = re.sub(r"[^ACGT]", "", seq)
        seq = seq[-self.seq_len:] if len(seq) >= self.seq_len else seq + "N" * (self.seq_len - len(seq))

        idxs = [base_map.get(b, -1) for b in seq]
        one_hot = torch.zeros(self.seq_len, 4, dtype=torch.float32)
        for pos, idx in enumerate(idxs):
            if idx >= 0:
                one_hot[pos, idx] = 1.0
        return one_hot
    
import torch
import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset
from scipy.stats import boxcox


class UtrOracleDataset(Dataset):
    def __init__(self, seq_len=50):
        
        csv_path='/root/autodl-tmp/graph_model/Diffusion_directed_evolution/5UTR/Random_UTRs.csv' 
        # 参数
        self.seq_len = seq_len
        
        # 读取CSV
        utr = pd.read_csv(csv_path)
        
        # 取序列和标签
        self.utrs = utr['UTR'].tolist()
        self.grs = utr['growth_rate'].values
        
        # --- 平移，使 growth_rate > 0 ---
        self.shift = -np.min(self.grs) + 1e-3
        self.grs = self.grs + self.shift
        
        # --- Box-Cox 变换 ---
        self.grs, self.lmbda = boxcox(self.grs)
        
    def __len__(self):
        return len(self.utrs)

    def __getitem__(self, idx):
        # 序列独热编码
        utr_onehot = self._str_to_one_hot(self.utrs[idx])
        
        # 标签值
        gr = torch.tensor(self.grs[idx], dtype=torch.float32)
        
        return utr_onehot, gr
    
    def _str_to_one_hot(self, seq: str) -> torch.FloatTensor:
        # ACGT 独热编码
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        # 去除非ACGT字符
        seq = re.sub(r"[^ACGT]", "", seq)
        
        # 右侧截断 / 填充
        if len(seq) >= self.seq_len:
            seq = seq[-self.seq_len:]   # 右侧截断
        else:
            seq = seq + "N" * (self.seq_len - len(seq))   # 填充N
        
        # 转换独热编码
        idxs = [base_map.get(b, -1) for b in seq]
        one_hot = torch.zeros(self.seq_len, 4, dtype=torch.float32)
        for pos, idx in enumerate(idxs):
            if idx >= 0:
                one_hot[pos, idx] = 1.0
        return one_hot


import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Optional
from sklearn.linear_model import LinearRegression


def kmers_stride1(seq: str, k: int = 6) -> List[str]:
    """将 DNA 序列按 1-bp 滑窗切成 k-mer 列表"""
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]

class SpeciesLMDataset(Dataset):
    def __init__(
        self,
    ):
        super().__init__()

        species = 'km'
        max_length = None  # None 表示使用 tokenizer 的默认最大长度
        proxy_species = "pichia_kudriavzevii_gca_000764455" # species token to use
        ## candidate species token name:
        # yarrowia_lipolytica, yarrowia_lipolytica_gca_001761485, yarrowia_lipolytica_gca_003367845, yarrowia_lipolytica_gca_014490615
        # kluyveromyces_marxianus, kluyveromyces_marxianus_dmku3_1042_gca_001417885, kluyveromyces_marxianus_gca_001417835
        # komagataella_pastoris, komagataella_pastoris_gca_001708105, komagataella_phaffii_cbs_7435_gca_000223565, komagataella_phaffii_gs115_gca_001746955
        # kazachstania_africana_cbs_2517_gca_000304475 for S. cerevisiae
        # IO: pichia_kudriavzevii_gca_000764455, pichia_kudriavzevii_gca_001983325, pichia_kudriavzevii_gca_002166775, pichia_kudriavzevii_gca_003054445
        
        #data_file_path = f'/root/autodl-tmp/graph_model/Diffusion_directed_evolution/SpeciesLM/data/{species}_dataset_with_promoters_500_and_codon_frequency.csv'
        data_file_path = f'/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/io_rna_seq/IO_dataset/io_dataset_with_promoters_and_codon_frequency.csv'
        data = pd.read_csv(data_file_path)
        
        # if species != 'sc':
        #     data = data[data['TPM']>5].dropna(subset=['promoter_sequence', 'gene_sequence'])
        #     data['TPM_bc_transformed'] = boxcox(np.array(data['TPM']), lmbda=0.17)  # 使用Box-Cox变换
        # else:
        #     data['TPM_bc_transformed'] = data['TPM_avg_transformed']
        #data['TPM'] = data[['r1', 'r2', 'r3']].mean(axis=1)
        data = data[data['TPM']>5].dropna(subset=['promoter_sequence', 'gene_sequence'])
        data['TPM_bc_transformed'] = boxcox(np.array(data['TPM']), lmbda=0.17)  # 使用Box-Cox变换

        self.labels = np.array(data['TPM_bc_transformed'])
        
        # codon_info = data['codon_count'].to_numpy().reshape(-1, 1)
        # linreg = LinearRegression().fit(codon_info, self.labels)
        # y_pred_cc = linreg.predict(codon_info)
        # self.labels   = self.labels - y_pred_cc
        
        self.labels = torch.tensor(self.labels, dtype=torch.float)

        # ---- 2. 初始化 tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision = "upstream_species_lm")
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        def tok_func_species(x, species_proxy, seq_col):
            res = self.tokenizer(species_proxy + " " +  " ".join(kmers_stride1(x[seq_col])))
            return res
        
        def tok_func_standard(x, seq_col): return self.tokenizer(" ".join(kmers_stride1(x[seq_col])))
        
        # ---- 3. 预处理序列：加 species token + k-mer 切片 ----
        species_token = f"[{proxy_species}]" if proxy_species else None
        enc_list = []
        for seq in data['promoter_sequence'].astype(str):
            # 组装单行 DataFrame-like dict 以复用作者函数
            row = {'promoter_sequence': seq.upper()}

            if species_token:
                enc = tok_func_species(row, species_token, 'promoter_sequence')
            else:
                enc = tok_func_standard(row, 'promoter_sequence')

            enc_list.append(enc)

        # 把列表中的 dict 拼成 batch 张量
        self.input_ids      = torch.stack(
            [torch.tensor(e["input_ids"]) for e in enc_list]
        )
        self.attention_mask = torch.stack(
            [torch.tensor(e["attention_mask"]) for e in enc_list]
        )

    # --------- Dataset API ---------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }
