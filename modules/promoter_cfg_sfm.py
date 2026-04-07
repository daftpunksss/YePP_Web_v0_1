import copy

import numpy as np
import pandas as pd
import torch, time, os
import swanlab
import yaml

from torch import optim
from torch.nn import functional as F

from utils.esm import upgrade_state_dict
# Remove the old flow utilities
# from utils.flow_utils import DirichletConditionalFlow, expand_simplex, sample_cond_prob_path, simplex_proj

from modules.general_module import GeneralModule
from utils.logging import get_logger
from model.promoter_model import PromoterModel
from utils.oracle_integrated import uas_infer, cp_infer, specieslm_sc_infer

# Add the SFM imports - you'll need to copy the categorical.py file from the SFM repo
from model.categorical import SphereCategoricalFlow

logger = get_logger(__name__)

class VFWrapper(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base  # 你的 PromoterModel / UNet

    def forward(self, pt, t, cond, cond_drop_mask=None):
        # 统一成 PromoterModel 的签名顺序
        return self.base(pt, cond, t=t, cond_drop_mask=cond_drop_mask)


class PromoterModule(GeneralModule):
    def __init__(self, args):
        super().__init__(args)

        self.model = PromoterModel(args)
        
        # Replace DirichletConditionalFlow with SphereCategoricalFlow
        self.sfm_flow = SphereCategoricalFlow(
            encoder=VFWrapper(self.model),
            data_dims=(500,),
            n_class=self.model.alphabet_size,
            ot=False,
            # ot_reg=1e-2,
            # momentum=0.0
        )
        
        self.loaded_distill_model = False
        self.condition_dropout_prob = 0.1
        
        self.best_val_loss = float('inf')

    def on_load_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {k: v for k,v in checkpoint['state_dict'].items() if 'distill_model' not in k}

    def training_step(self, batch, batch_idx):
        self.stage = 'train'
        loss = self.general_step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        if self.args.validate:
            self.try_print_log()
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.trainer.save_checkpoint(os.path.join(os.environ["MODEL_DIR"], f"epoch={self.trainer.current_epoch}-step={self.trainer.global_step}-val_loss={self.best_val_loss}.ckpt"))

    def general_step(self, batch, batch_idx=None):
        self.iter_step += 1
        seq_one_hot, codon_info, species = batch
        cond_full = torch.cat([codon_info, species], dim=-1) 
        seq = torch.argmax(seq_one_hot, dim=-1)
        B, L = seq.shape
        
        # Condition dropout
        drop_mask = (torch.rand(B, device=self.device) < self.condition_dropout_prob)
        cond_in = cond_full.clone()
        cond_in[drop_mask] = 0

        if self.args.mode in ('sfm', 'statistical'):
            # ======= SFM: Flow-Matching（Riemannian MSE on velocity field）=======
            K = self.model.alphabet_size
            B = seq.size(0)

            # 目标分布 x1（one-hot），基分布 x0（由 flow 内部采样）
            x1 = F.one_hot(seq, num_classes=K).float()                           # [B, L, K]

            # --- 按 categorical.py 的 get_loss 逻辑自己展开一遍，便于记录 t 和逐样本 loss ---
            # 1) 采样噪声（基分布）与时间
            noise = self.sfm_flow.sample_prior(B, self.sfm_flow.total_data_dim,
                                               self.sfm_flow.n_class, device=self.device)  # [B, L, K]
            t = torch.rand(B, device=self.device) * self.sfm_flow.max_t                   # (B,)

            # 2) 若开启 OT，则做 batch 级最优匹配（与 categorical.py 一致）
            cond_in_eff, drop_mask_eff = cond_in, drop_mask
            if self.sfm_flow.ot:
                noise, x1, (cond_in_eff, drop_mask_eff) = self.sfm_flow.batch_ot(
                    noise, x1, (cond_in, drop_mask)
                )

            # 3) 沿统计流形的 geodesic 构造 (pt, vf_target)
            pt, vf_target = self.sfm_flow.vecfield(
                self.sfm_flow.preprocess(noise),
                self.sfm_flow.preprocess(x1),
                t[:, None],
                self.sfm_flow.eps
            )  # pt, vf_target: [B, L, K]

            # 4) 预测矢量场（注意：SFM.forward 的签名是 (t, pt, *cond_args)）
            vf_pred = self.sfm_flow(t, pt, cond_in_eff, drop_mask_eff)         

            per_pos_err = self.sfm_flow.norm2(pt, vf_pred - vf_target, self.sfm_flow.eps) 
            losses = per_pos_err.mean(dim=-1)    
            
        elif self.args.mode == 'ardm' or self.args.mode == 'lrar':
            mask_prob = torch.rand(1, device=self.device)
            mask = torch.rand(seq.shape, device=self.device) < mask_prob
            if self.args.mode == 'lrar': mask = ~(torch.arange(L, device=self.device) < (1-mask_prob) * L)
            xt = torch.where(mask, 4, seq) # mask token has idx 4
            xt = torch.nn.functional.one_hot(xt, num_classes=5)
            alphas = mask_prob.expand(B)
            logits = self.model(xt, cond_in, t=alphas, cond_drop_mask=drop_mask)
            losses = F.cross_entropy(logits.transpose(1, 2), seq, reduction='none').mean(-1)
            
        elif self.args.mode == 'distill':
            if self.stage == 'val':
                seq_distill = torch.zeros_like(seq, device=self.device)
                xt = torch.ones((B,L, self.model.alphabet_size), device=self.device)
                xt = xt / xt.sum(-1)[..., None]
            else:
                logits_distill, xt = self.sfm_flow_inference(seq, codon_info, model=self.distill_model, args=self.distill_args)
                seq_distill = torch.argmax(logits_distill, dim=-1)
            alphas = torch.zeros(B, device=self.device)
            logits = self.model(xt, cond_in, t=alphas, cond_drop_mask=drop_mask)
            losses = F.cross_entropy(logits.transpose(1, 2), seq_distill if self.args.mode == 'distill' else seq, reduction='none').mean(-1)

        self.lg('alpha' if self.args.mode != 'sfm' else 't', alphas if self.args.mode != 'sfm' else t)
        self.lg('loss', losses)
        self.lg('perplexity', torch.exp(losses.mean())[None].expand(B))
        self.lg('dur', torch.tensor(time.time() - self.last_log_time)[None].expand(B))
        
        if self.stage == "val":
            cond_in_rounded = torch.round(cond_in * 100) / 100
            #self.lg('condition_input', cond_in_rounded)
            
            if self.args.mode == 'sfm' or self.args.mode == 'statistical':
                # SFM inference with guidance
                logits_pred = self.sfm_flow_inference_with_guidance(seq, cond_full)
                seq_pred = torch.argmax(logits_pred, dim=-1)
                
            elif self.args.mode == 'ardm' or self.args.mode == 'lrar':
                seq_pred = self.ar_inference(seq, codon_info)
                
            elif self.args.mode == 'distill':
                logits_pred = self.distill_inference(seq, codon_info)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            else:
                raise NotImplementedError()
            
            self.lg('generated_seq', [''.join([['A','C','G','T'][num] for num in seq]) for seq in seq_pred])
            self.lg('ori_seq', [''.join([['A','C','G','T'][num] for num in seq]) for seq in seq])
            self.lg('recovery', seq_pred.eq(seq).float().mean(-1))
            
            # if self.args.mode == 'sfm' or self.args.mode == 'statistical':
            #     # For logging purposes, compute unconditional and conditional separately
            #     logits_uncond = self.sfm_flow_inference(seq, torch.zeros_like(cond_full))
            #     logits_cond = self.sfm_flow_inference(seq, cond_full)
            #     self.lg('recovery_uncond', torch.argmax(logits_uncond, -1).eq(seq).float().mean(-1))
            #     self.lg('recovery_cond', torch.argmax(logits_cond, -1).eq(seq).float().mean(-1))
            
            self.lg('guidance_scale', torch.tensor([self.args.guidance_scale]*B, device=self.device))
            
        self.last_log_time = time.time()
        return losses.mean()

    @torch.no_grad()
    def sfm_flow_inference(self, seq, emb):
        """
        Unguided SFM inference using the built-in sampler (geometry-consistent).
        """
        B, L = seq.shape
        K = self.model.alphabet_size
        device = seq.device

        # 1) 采样终点概率（和训练完全一致的几何 & 步进）
        method = 'euler'
        n_steps = getattr(self.args, 'num_integration_steps', 300)

        xt = self.sfm_flow.sample(
            method=method,
            n_sample=B,
            n_steps=n_steps,
            device=device,
            *[emb]  # 只传 cond；无 drop_mask
        )  # -> [B, L, K] on simplex (postprocessed)

        # 2) 最终 logits（你原本就是这么做的）
        logits = self.model(xt, emb, t=torch.ones(B, device=device))
        return logits


    @torch.no_grad()
    def sfm_flow_inference_with_guidance(self, seq, cond_full):
        B, L = seq.shape
        K = self.model.alphabet_size
        device = seq.device

        num_steps = getattr(self.args, 'num_integration_steps', 300)
        ts = torch.linspace(0., 1., num_steps + 1, device=device)      # 统一时间网格
        dt = ts[1] - ts[0]

        # 轻微破对称起点（可选）：避免样本坍缩
        # x0 = self.sfm_flow.sample_prior(B, L, K, device=device)       # Dirichlet 噪声
        x0 = torch.full((B, L, K), 1.0 / K, device=device)
        xt = self.sfm_flow.preprocess(x0)                               # 和采样器一致

        for i in range(num_steps):
            t = ts[i]
            t_batch = torch.full((B,), t.item(), device=device)

            vf_uncond = self.sfm_flow.forward(
                t_batch, xt, torch.zeros_like(cond_full),
                torch.ones(B, dtype=torch.bool, device=device)
            )
            vf_cond = self.sfm_flow.forward(
                t_batch, xt, cond_full,
                torch.zeros(B, dtype=torch.bool, device=device)
            )
            vf = vf_uncond + self.args.guidance_scale * (vf_cond - vf_uncond)

            xt = self.sfm_flow.exp(xt, vf * dt, self.sfm_flow.eps)
            xt = self.sfm_flow.proj_x(xt)                               # 一步投影，和采样器一致

        xt = self.sfm_flow.postprocess(xt)                               # 回到 simplex 概率
        logits = self.model(xt, cond_full, t=torch.ones(B, device=device))
        return logits


    @torch.no_grad()
    def sfm_flow_inference_distill(self, seq, emb, model=None, args=None):
        """
        SFM inference that can be used with distillation
        """
        if model is None:
            model = self.model
        if args is None:
            args = self.args
            
        B, L = seq.shape
        K = model.alphabet_size
        
        # Start from uniform distribution
        x0 = torch.ones(B, L, K, device=seq.device) / K
        
        num_steps = getattr(args, 'num_integration_steps', 100)
        dt = 1.0 / num_steps
        
        xt = x0
        for step in range(num_steps):
            t = step * dt
            t_batch = torch.full((B,), t, device=seq.device)
            
            # 用 SFM 直接预测速度场（矢量场），再用指数映射在流形上走一步
            vf = self.sfm_flow.forward(t_batch, xt, emb)       # 速度场 v_t(x)
            xt = self.sfm_flow.exp(xt, vf * dt)                # x_{t+dt} = Exp_x (v*dt)

        # Get final logits
        logits = model(xt, emb, t=torch.ones(B, device=seq.device))
        
        return logits, x0

    @torch.no_grad()
    def distill_inference(self, seq, emb):
        B, L = seq.shape
        K = self.model.alphabet_size
        x0 = torch.ones(B, L, K, device=seq.device) / K
        logits = self.model(x0, emb, t=torch.zeros(B, device=self.device))
        return logits

    @torch.no_grad()
    def ar_inference(self, seq, emb):
        B, L = seq.shape
        order = np.arange(L)
        if self.args.mode =='ardm': np.random.shuffle(order)
        curr = (torch.ones((B, L), device=self.device) * 4).long()
        for i, k in enumerate(order):
            t = torch.tensor( i/L ,device=self.device)
            logits = self.model(torch.nn.functional.one_hot(curr, num_classes=5), emb, t[None].expand(B))
            curr[:, k] = torch.distributions.Categorical(probs=torch.nn.functional.softmax(logits[:, k] / self.args.flow_temp, -1)).sample()
        return curr

    @torch.no_grad()
    def on_validation_epoch_start(self) -> None:
        self.generator = np.random.default_rng(seed=137)

    def load_distill_model(self):
        with open(self.args.distill_ckpt_hparams) as f:
            hparams = yaml.load(f, Loader=yaml.UnsafeLoader)
            self.distill_args = copy.deepcopy(hparams['args'])
        self.distill_model = PromoterModel(self.distill_args)
        upgraded_dict = upgrade_state_dict(torch.load(self.args.distill_ckpt, map_location=self.device)['state_dict'], prefixes=['model.'])
        self.distill_model.load_state_dict(upgraded_dict)
        self.distill_model.eval()
        self.distill_model.to(self.device)
        for param in self.distill_model.parameters():
            param.requires_grad = False

    def on_train_epoch_start(self) -> None:
        if not self.loaded_distill_model and self.args.distill_ckpt is not None:
            self.load_distill_model()
            self.loaded_distill_model = True

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        self.generator = np.random.default_rng()
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
        print({k: len(v) for k, v in log.items()})
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update({'epoch': self.trainer.current_epoch, 'step': self.trainer.global_step, 'iter_step': self.iter_step})

        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            self.log_dict(mean_log, batch_size=1)
            if self.args.swanlab:
                swanlab.log(mean_log)

            path = os.path.join(os.environ["MODEL_DIR"], f"val_{self.trainer.global_step}.csv")
            pd.DataFrame(log).to_csv(path)

        for key in list(log.keys()):
            if "val_" in key:
                del self._log[key]

    def lg(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        log = self._log
        if self.args.validate or self.stage == 'train':
            log["iter_" + key].extend(data)
        log[self.stage + "_" + key].extend(data)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer