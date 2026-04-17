import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
import argparse
import time
from tqdm import tqdm
import pandas as pd

# 需要导入的工具函数
from utils.flow_utils import DirichletConditionalFlow, expand_simplex, simplex_proj
from model.promoter_model import PromoterModel

class DirichletPromoterInference:
    """
    使用训练好的Dirichlet模型进行DNA序列推断
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 device: str = 'cuda:0',
                 args: Optional[argparse.Namespace] = None):
        """
        初始化推断器
        """
        self.device = device
        self.args = args if args is not None else self._get_default_args()
        
        # 加载模型
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # 初始化条件流
        self.condflow = DirichletConditionalFlow(
            K=self.model.alphabet_size, 
            alpha_spacing=0.01, 
            alpha_max=self.args.alpha_max
        )
        
        # DNA碱基映射
        self.alphabet = ['A', 'C', 'G', 'T']
        self.alphabet_size = 4
        
    def _get_default_args(self):
        """获取默认参数"""
        args = argparse.Namespace()
        args.mode = 'dirichlet'
        args.prior_pseudocount = 2.0
        args.alpha_max = 100.0
        args.guidance_scale = 1.0
        args.num_integration_steps = 400
        args.flow_temp = 1.0
        return args
        
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """加载模型"""
        # 创建模型实例
        model = PromoterModel(self.args).to(self.device)
        
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)
            
        return model
    
    @torch.no_grad()
    def dirichlet_flow_inference(self, 
                                codon_info: torch.Tensor,
                                species: torch.Tensor,
                                guidance_scale: float = 1.0,
                                batch_size: int = 1,
                                seq_length: int = 500,
                                use_optimized_impl: bool = True,
                                enable_profile: bool = False,
                                enable_step_checks: bool = True,
                                check_every_n_steps: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用Dirichlet flow进行序列生成
        
        参数:
        codon_info : torch.Tensor
            密码子信息 [B, codon_dim]
        species : torch.Tensor
            物种信息 [B, species_dim]
        guidance_scale : float
            引导尺度
        batch_size : int
            批次大小
        seq_length : int
            序列长度
            
        返回:
        generated_seq : torch.Tensor
            生成的序列 [B, L]
        x0 : torch.Tensor
            初始分布
        """
        B = batch_size
        L = seq_length
        K = self.alphabet_size
        
        # 合并条件信息
        cond_full = torch.cat([codon_info, species], dim=-1)
        cond_zeros = torch.zeros_like(cond_full)
        
        # 初始化：从Dirichlet分布采样
        x0 = torch.distributions.Dirichlet(
            torch.ones(B, L, K, device=self.device)
        ).sample()
        
        # 单位矩阵用于计算条件流
        eye = torch.eye(K).to(self.device)
        
        # 当前状态
        xt = x0.clone()
        cond_drop_true = torch.ones(B, dtype=torch.bool, device=self.device)
        cond_drop_false = torch.zeros(B, dtype=torch.bool, device=self.device)
        cond_drop_cfg = torch.cat([cond_drop_true, cond_drop_false], dim=0)
        simplex_target = torch.ones((B, L), device=self.device)
        timings = {
            "model_forward_time": 0.0,
            "c_factor_time": 0.0,
            "sampling_update_time": 0.0,
            "checks_time": 0.0,
        }
        c_factor_recorder = None
        if enable_profile:
            def c_factor_recorder(duration: float) -> None:
                timings["c_factor_time"] += duration
        
        # 时间步：从1到alpha_max
        t_span = torch.linspace(1, self.args.alpha_max, 
                               self.args.num_integration_steps, 
                               device=self.device)
        
        # 积分过程
        for i, (s, t) in enumerate(tqdm(
            zip(t_span[:-1], t_span[1:]), 
            total=len(t_span)-1, 
            desc="Generating sequence"
        )):
            # 计算先验权重
            prior_weight = self.args.prior_pseudocount / (s + self.args.prior_pseudocount - 1)
            
            # 扩展simplex（与训练时一致）
            seq_xt = torch.cat([
                xt * (1 - prior_weight), 
                xt * prior_weight
            ], -1)
            
            # 时间张量
            t_tensor = s[None].expand(B)
            
            if enable_profile and self.device.startswith("cuda"):
                torch.cuda.synchronize(device=self.device)
                forward_start = time.perf_counter()
            if guidance_scale != 0:
                if use_optimized_impl:
                    logits_2b = self.model(
                        torch.cat([seq_xt, seq_xt], dim=0),
                        torch.cat([cond_zeros, cond_full], dim=0),
                        t=s[None].expand(2 * B),
                        cond_drop_mask=cond_drop_cfg,
                    )
                    logits_uncond, logits_cond = logits_2b.chunk(2, dim=0)
                else:
                    logits_uncond = self.model(
                        seq_xt,
                        cond_zeros,
                        t=t_tensor,
                        cond_drop_mask=cond_drop_true,
                    )
                    logits_cond = self.model(
                        seq_xt,
                        cond_full,
                        t=t_tensor,
                        cond_drop_mask=cond_drop_false,
                    )
                logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
            else:
                logits = self.model(
                    seq_xt,
                    cond_zeros,
                    t=t_tensor,
                    cond_drop_mask=cond_drop_true,
                )
            if enable_profile and self.device.startswith("cuda"):
                torch.cuda.synchronize(device=self.device)
                timings["model_forward_time"] += time.perf_counter() - forward_start
            
            # 计算输出概率
            out_probs = torch.nn.functional.softmax(logits / self.args.flow_temp, -1)
            
            # 计算c_factor
            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item(), profile_recorder=c_factor_recorder)
            c_factor = torch.from_numpy(c_factor).to(xt)
            
            # 处理NaN
            if torch.isnan(c_factor).any():
                print(f'Warning: NaN in c_factor, replacing with 0')
                c_factor = torch.nan_to_num(c_factor)
            
            # 计算条件流
            if enable_profile and self.device.startswith("cuda"):
                torch.cuda.synchronize(device=self.device)
                update_start = time.perf_counter()
            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            
            # 计算流
            flow = (out_probs.unsqueeze(-2) * cond_flows).sum(-1)
            
            # 更新xt
            xt = xt + flow * (t - s)
            if enable_profile and self.device.startswith("cuda"):
                torch.cuda.synchronize(device=self.device)
                timings["sampling_update_time"] += time.perf_counter() - update_start
            
            # 确保xt在simplex上
            should_check = enable_step_checks and (i % check_every_n_steps == 0)
            if should_check:
                if enable_profile and self.device.startswith("cuda"):
                    torch.cuda.synchronize(device=self.device)
                    check_start = time.perf_counter()
                if not torch.allclose(xt.sum(2), simplex_target, atol=1e-4) or not (xt >= 0).all():
                    xt = simplex_proj(xt)
                if enable_profile and self.device.startswith("cuda"):
                    torch.cuda.synchronize(device=self.device)
                    timings["checks_time"] += time.perf_counter() - check_start
        
        # 最终序列：取argmax
        generated_seq = torch.argmax(xt, dim=-1)
        
        return generated_seq, x0
    
    def generate_sequences(self,
                         conditions: List[Tuple[torch.Tensor, torch.Tensor]],
                         num_sequences_per_condition: int = 1,
                         **kwargs) -> List[str]:
        """
        批量生成序列
        """
        all_sequences = []
        
        for codon_info, species in conditions:
            # 确保输入在正确的设备上
            codon_info = codon_info.to(self.device)
            species = species.to(self.device)
            
            # 如果需要生成多个序列，复制条件
            if num_sequences_per_condition > 1:
                codon_info = codon_info.repeat(num_sequences_per_condition, 1)
                species = species.repeat(num_sequences_per_condition, 1)
            
            # 生成序列
            generated_seq, _ = self.dirichlet_flow_inference(
                codon_info=codon_info,
                species=species,
                batch_size=codon_info.shape[0],
                guidance_scale=self.args.guidance_scale,
            )
            
            # 转换为字符串
            for seq in generated_seq:
                seq_str = ''.join([self.alphabet[idx] for idx in seq.cpu().numpy()])
                all_sequences.append(seq_str)
        
        return all_sequences
    
    def visualize_generation_process(self, 
                                   codon_info: torch.Tensor,
                                   species: torch.Tensor,
                                   save_path: str = 'generation_process.png',
                                   num_snapshots: int = 10):
        """
        可视化生成过程，保存中间状态
        """
        import matplotlib.pyplot as plt
        
        B = 1  # 只可视化一个序列
        L = 500
        K = self.alphabet_size
        
        # 合并条件
        cond_full = torch.cat([codon_info[:1], species[:1]], dim=-1)
        
        # 初始化
        x0 = torch.distributions.Dirichlet(
            torch.ones(B, L, K, device=self.device)
        ).sample()
        
        eye = torch.eye(K).to(self.device)
        xt = x0.clone()
        
        # 时间步
        t_span = torch.linspace(1, self.args.alpha_max, 
                               self.args.num_integration_steps, 
                               device=self.device)
        
        # 保存快照的步骤
        snapshot_steps = np.linspace(0, len(t_span)-2, num_snapshots, dtype=int)
        snapshots = []
        
        # 积分过程
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            # 与上面相同的推断过程
            prior_weight = self.args.prior_pseudocount / (s + self.args.prior_pseudocount - 1)
            seq_xt = torch.cat([xt * (1 - prior_weight), xt * prior_weight], -1)
            t_tensor = s[None].expand(B)
            
            # 计算logits
            logits_uncond = self.model(
                seq_xt, torch.zeros_like(cond_full), t=t_tensor,
                cond_drop_mask=torch.ones(B, dtype=torch.bool, device=self.device)
            )
            logits_cond = self.model(
                seq_xt, cond_full, t=t_tensor,
                cond_drop_mask=torch.zeros(B, dtype=torch.bool, device=self.device)
            )
            logits = logits_uncond + self.args.guidance_scale * (logits_cond - logits_uncond)
            
            # 更新xt
            out_probs = torch.nn.functional.softmax(logits / self.args.flow_temp, -1)
            c_factor = self.condflow.c_factor(xt.detach().cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)
            c_factor = torch.nan_to_num(c_factor)
            
            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (out_probs.unsqueeze(-2) * cond_flows).sum(-1)
            xt = xt + flow * (t - s)
            xt = simplex_proj(xt)
            
            # 保存快照
            if i in snapshot_steps:
                snapshots.append({
                    'step': i,
                    'time': s.item(),
                    'probs': xt[0].detach().cpu().numpy(),  # [L, K]
                    'seq': torch.argmax(xt[0], dim=-1).detach().cpu().numpy()
                })
        
        # 创建可视化
        fig, axes = plt.subplots(num_snapshots, 1, figsize=(20, 2*num_snapshots))
        if num_snapshots == 1:
            axes = [axes]
        
        for idx, snapshot in enumerate(snapshots):
            # 显示概率分布
            im = axes[idx].imshow(snapshot['probs'][:100, :].T,  # 只显示前100个位置
                                 aspect='auto', cmap='hot')
            axes[idx].set_ylabel(f't={snapshot["time"]:.1f}')
            axes[idx].set_yticks(range(4))
            axes[idx].set_yticklabels(self.alphabet)
            
            if idx == len(snapshots) - 1:
                axes[idx].set_xlabel('Position')
        
        plt.colorbar(im, ax=axes, label='Probability')
        plt.suptitle('DNA Sequence Generation Process (Dirichlet Flow)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {save_path}")
        
        # 返回最终序列
        final_seq = ''.join([self.alphabet[idx] for idx in snapshots[-1]['seq']])
        return final_seq, snapshots


import pandas as pd
import torch
import argparse
import os
from datetime import datetime

# 使用示例
if __name__ == "__main__":
    # 设置参数
    checkpoint_path = "/root/autodl-tmp/graph_model/Diffusion_directed_evolution/workdir/500bp_promoter_all_speceis_cfg_gs1_2025-07-19_21-40-10/epoch=104-step=42315-val_loss=0.6868324279785156.ckpt"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 创建参数
    args = argparse.Namespace()
    args.mode = 'dirichlet'
    args.prior_pseudocount = 2.0
    args.alpha_max = 8.0
    args.num_integration_steps = 400
    args.flow_temp = 1.0
    args.guidance_scale = 1.0  # 可以调整这个值来控制条件的强度
    # args.target_gene_ids = [
    #                         "YGR192C","YPR080W","YLR110C","YKL060C","YHR174W","YOL086C",
    #                         "YMR094W","YMR063W","YNR063W","YBR184W","YOR378W","YPR146C",
    #                         ]  # 示例基因ID列表
    lk = pd.read_csv('/root/autodl-tmp/graph_model/Diffusion_directed_evolution/SpeciesLM/AAA_plotting_code_and_file/cfg_promoters_evaluation/codon_sym/codon_ga.csv')
    args.target_gene_ids = lk['gene_id']
    # 准备数据
    lian_lab_data_path = "/root/autodl-tmp/graph_model/Diffusion_directed_evolution/SpeciesLM/cross_species_data/lianlab_aval_yeast_promoter_gene_info.csv"
    lian_lab_data = pd.read_csv(lian_lab_data_path)
    data_file_path = "/root/autodl-tmp/graph_model/Diffusion_directed_evolution/SpeciesLM/AAA_plotting_code_and_file/cfg_promoters_evaluation/codon_sym/codon_ga.csv"
    data = pd.read_csv(data_file_path)

    # codon 列名（64 个）
    CODON_COLS = [
        'AAA','AAC','AAG','AAT','ACA','ACC','ACG','ACT','AGA','AGC','AGG','AGT',
        'ATA','ATC','ATG','ATT','CAA','CAC','CAG','CAT','CCA','CCC','CCG','CCT',
        'CGA','CGC','CGG','CGT','CTA','CTC','CTG','CTT','GAA','GAC','GAG','GAT',
        'GCA','GCC','GCG','GCT','GGA','GGC','GGG','GGT','GTA','GTC','GTG','GTT',
        'TAA','TAC','TAG','TAT','TCA','TCC','TCG','TCT','TGA','TGC','TGG','TGT',
        'TTA','TTC','TTG','TTT'
    ]

    # 过滤数据
    # data = data.dropna(subset=['promoter_sequence', 'gene_sequence', *CODON_COLS])
    # data = data[data['promoter_sequence'].str.len() == 500].reset_index(drop=True)

    # 物种映射
    species_unique = sorted(lian_lab_data["species"].unique())
    _sp2idx = {sp: i for i, sp in enumerate(species_unique)}
    species_dim = len(species_unique)  # 使用实际的物种数量
    
    print(f"Found {species_dim} unique species")
    print(f"Species list: {species_unique[:5]}..." if len(species_unique) > 5 else f"Species list: {species_unique}")
    
    # 初始化推断器
    print("\nInitializing model...")
    inferencer = DirichletPromoterInference(
        checkpoint_path=checkpoint_path,
        device=device,
        args=args
    )
    
    # 准备条件
    conditions = []
    gene_info = []  # 保存基因信息用于后续输出
    
    print("\nPreparing conditions...")
    for gid in args.target_gene_ids:
        row = data.loc[data["gene_id"] == gid]
        if row.empty:
            print(f"[WARNING] gene_id '{gid}' not found in filtered DataFrame, skipping...")
            continue
        
        # 获取第一行数据
        row_data = row.iloc[0]
        
        # 64 维密码子频率向量
        codon_vec = torch.tensor(
            row_data[CODON_COLS].to_numpy(dtype="float32"),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)  # 形状 [1, 64]
        
        # one-hot 物种向量
        sp_idx = _sp2idx[row_data["species"]]
        species_vec = torch.nn.functional.one_hot(
            torch.tensor([sp_idx], device=device),
            num_classes=species_dim,
        ).to(torch.float32)  # 形状 [1, species_dim]
        
        conditions.append((codon_vec, species_vec))
        gene_info.append({
            'gene_id': gid,
            'species': row_data["species"],
            #'original_promoter': row_data["promoter_sequence"]
        })
        
        print(f"  Added condition for gene {gid} from species {row_data['species']}")
    
    if not conditions:
        raise ValueError("No valid gene IDs found in the dataset!")
    
    # 设置批次大小
    batch_size = 2000
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/root/autodl-tmp/graph_model/Diffusion_directed_evolution/cfg_dfm_generated_results/dfm_generated_sequences_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成序列
    print(f"\nGenerating {batch_size} sequences for each of {len(conditions)} genes...")
    sequences = inferencer.generate_sequences(
        conditions=conditions,
        num_sequences_per_condition=batch_size
    )
    
    print(f"Total sequences generated: {len(sequences)}")
    
    # 保存所有序列到主FASTA文件
    main_fasta_path = os.path.join(output_dir, 'all_generated_sequences.fasta')
    with open(main_fasta_path, 'w') as f:
        for gene_idx, info in enumerate(gene_info):
            start_idx = gene_idx * batch_size
            end_idx = start_idx + batch_size
            
            for seq_idx, seq in enumerate(sequences[start_idx:end_idx]):
                f.write(f">Generated_{info['gene_id']}_seq{seq_idx + 1:03d}_species_{info['species']}\n")
                f.write(f"{seq}\n")
    
    print(f"All sequences saved to {main_fasta_path}")
    
    # 为每个基因创建单独的文件
    for gene_idx, info in enumerate(gene_info):
        gene_file = os.path.join(output_dir, f"{info['gene_id']}_generated_promoters.fasta")
        with open(gene_file, 'w') as f:
            start_idx = gene_idx * batch_size
            end_idx = start_idx + batch_size
            
            for seq_idx, seq in enumerate(sequences[start_idx:end_idx]):
                f.write(f">Generated_{info['gene_id']}_seq{seq_idx + 1:03d}_species_{info['species']}\n")
                f.write(f"{seq}\n")
        
        print(f"  Saved {batch_size} sequences for gene {info['gene_id']} to {gene_file}")
    
    # 创建摘要文件
    summary_path = os.path.join(output_dir, 'generation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("DNA Sequence Generation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generation timestamp: {timestamp}\n")
        f.write(f"Model checkpoint: {checkpoint_path}\n")
        f.write(f"Guidance scale: {args.guidance_scale}\n")
        f.write(f"Number of integration steps: {args.num_integration_steps}\n")
        f.write(f"Alpha max: {args.alpha_max}\n")
        f.write(f"Prior pseudocount: {args.prior_pseudocount}\n")
        f.write(f"Flow temperature: {args.flow_temp}\n\n")
        
        f.write(f"Total genes processed: {len(gene_info)}\n")
        f.write(f"Sequences per gene: {batch_size}\n")
        f.write(f"Total sequences generated: {len(sequences)}\n\n")
        
        for info in gene_info:
            f.write(f"\nGene: {info['gene_id']}\n")
            f.write(f"Species: {info['species']}\n")
            f.write(f"Original promoter (first 100bp): {info['original_promoter'][:100]}...\n")
    
    print(f"\nSummary saved to {summary_path}")
    
    # 创建CSV文件记录生成的序列信息
    sequences_df = []
    for gene_idx, info in enumerate(gene_info):
        start_idx = gene_idx * batch_size
        end_idx = start_idx + batch_size
        
        for seq_idx, seq in enumerate(sequences[start_idx:end_idx]):
            sequences_df.append({
                'gene_id': info['gene_id'],
                'species': info['species'],
                'sequence_id': f"seq{seq_idx + 1:03d}",
                'sequence': seq,
                'length': len(seq),
                'gc_content': (seq.count('G') + seq.count('C')) / len(seq)
            })
    
    df = pd.DataFrame(sequences_df)
    csv_path = os.path.join(output_dir, 'generated_sequences_info.csv')
    df.to_csv(csv_path, index=False)
    print(f"Sequence information saved to {csv_path}")
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("Generation Statistics:")
    print("=" * 50)
    for gene_idx, info in enumerate(gene_info):
        gene_df = df[df['gene_id'] == info['gene_id']]
        print(f"\nGene {info['gene_id']} ({info['species']}):")
        print(f"  - Sequences generated: {len(gene_df)}")
        print(f"  - Average GC content: {gene_df['gc_content'].mean():.3f}")
        print(f"  - GC content range: [{gene_df['gc_content'].min():.3f}, {gene_df['gc_content'].max():.3f}]")
    
    # # 可视化第一个基因的生成过程（可选）
    # if conditions and len(conditions) > 0:
    #     print("\nGenerating visualization for the first gene...")
    #     codon_info, species = conditions[0]
        
    #     viz_path = os.path.join(output_dir, f'generation_process_{gene_info[0]["gene_id"]}.png')
    #     final_seq, snapshots = inferencer.visualize_generation_process(
    #         codon_info=codon_info,
    #         species=species,
    #         save_path=viz_path,
    #         num_snapshots=10
    #     )
    #     print(f"Visualization saved to {viz_path}")
    
    # print(f"\n✅ All files saved to directory: {output_dir}")
    # print("\nGeneration complete!")
