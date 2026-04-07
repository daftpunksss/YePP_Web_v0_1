import copy

import numpy as np
import pandas as pd
import torch, time, os
import swanlab
import yaml

from torch import optim

from utils.esm import upgrade_state_dict
from utils.flow_utils import DirichletConditionalFlow, expand_simplex, sample_cond_prob_path, simplex_proj
from modules.general_module import GeneralModule
from utils.logging import get_logger
from model.promoter_model import PromoterModel


from utils.fisher import sphere_map, inv_sphere_map, proj_tangent, slerp, logmap_sphere


logger = get_logger(__name__)


class PromoterModule(GeneralModule):
    def __init__(self, args):
        super().__init__(args)

        self.model = PromoterModel(args)
        self.condflow = DirichletConditionalFlow(K=self.model.alphabet_size, alpha_spacing=0.01, alpha_max=args.alpha_max)
        self.loaded_distill_model = False
        self.condition_dropout_prob = 0.1
        self.fisher_prior = 'uniform'  # 'random' or 'uniform'
        
        # self.uas_oracle = uas_infer
        # self.cp_oracle = cp_infer
        
        self.best_val_loss = float('inf')

    def on_load_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {k: v for k,v in checkpoint['state_dict'].items() if 'distill_model' not in k}

    def training_step(self, batch, batch_idx):
        self.stage = 'train'
        loss = self.general_step(batch, batch_idx)
        # if self.args.ckpt_iterations is not None and self.trainer.global_step in self.args.ckpt_iterations:
        #     self.trainer.save_checkpoint(os.path.join(os.environ["MODEL_DIR"],f"epoch={self.trainer.current_epoch}-step={self.trainer.global_step}.ckpt"))
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
        #print(f'batch:{batch}')
        seq_one_hot, codon_info, species = batch
        cond_full = torch.cat([codon_info, species], dim=-1) 
        seq = torch.argmax(seq_one_hot, dim=-1)
        #print(f'seq_shape:{seq.shape}')
        B, L = seq.shape
        
                                        # 新增模式
        drop_mask = (torch.rand(B, device=self.device)               # 1 = 无条件
                    < self.condition_dropout_prob)
        cond_in = cond_full.clone()
        cond_in[drop_mask] = 0                                       # 置零 ⇒ 无条件

        if self.args.mode == 'dirichlet' or self.args.mode == 'riemannian':
            xt, alphas = sample_cond_prob_path(self.args, seq, self.model.alphabet_size)
            xt, prior_weights = expand_simplex(xt, alphas, self.args.prior_pseudocount)
            self.lg('prior_weight', prior_weights)
        elif self.args.mode == 'fisher':
            K = self.model.alphabet_size
            # 真值 one-hot -> simplex -> sphere
            y = torch.nn.functional.one_hot(seq, num_classes=K).float()
            x1 = sphere_map(y)

            # 选择源分布 p1：统一、稳定的做法是“均匀球面常量向量”
            # 也可用 Gamma 采样构建 S_+^{K-1} 上的随机点，实验对比可加 ablation
            if self.fisher_prior == 'uniform':
                x0 = torch.full_like(x1, 1.0 / np.sqrt(K))
            else:
                g = torch.distributions.Gamma(1.0, 1.0).sample(y.shape).to(y)
                x0 = sphere_map(g / g.sum(-1, keepdim=True))

            # 采样 t ∼ U(0,1)，按测地线插值得到 xt
            alphas = torch.rand(B, device=self.device)      # 复用你的日志键 'alpha'
            xt = slerp(x0, x1, alphas)

            # 目标向量场 ut = log_{xt}(x1)/(1 - t)
            ut = logmap_sphere(xt, x1) / (1.0 - alphas).view(-1,1,1).clamp_min(1e-4)

            # 模型直接回归向量场（不要 softmax），输出投影到切空间
            v = self.model(xt, cond_in, t=alphas, cond_drop_mask=drop_mask)
            v = proj_tangent(xt, v)

            # CFM 损失（切空间上的二范数）
            losses = ((v - ut) ** 2).sum(-1).mean(-1)

        elif self.args.mode == 'ardm' or self.args.mode == 'lrar':
            mask_prob = torch.rand(1, device=self.device)
            mask = torch.rand(seq.shape, device=self.device) < mask_prob
            if self.args.mode == 'lrar': mask = ~(torch.arange(L, device=self.device) < (1-mask_prob) * L)
            xt = torch.where(mask, 4, seq) # mask token has idx 4
            xt = torch.nn.functional.one_hot(xt, num_classes=5)
            alphas = mask_prob.expand(B)
        elif self.args.mode == 'distill':
            if self.stage == 'val':
                seq_distill = torch.zeros_like(seq, device=self.device)
                xt = torch.ones((B,L, self.model.alphabet_size), device=self.device)
                xt = xt / xt.sum(-1)[..., None]
            else:
                logits_distill, xt = self.dirichlet_flow_inference(seq, codon_info, model=self.distill_model, args=self.distill_args)
                seq_distill = torch.argmax(logits_distill, dim=-1)
            alphas = torch.zeros(B, device=self.device)

        # if not msa:
        logits = self.model(
            xt, cond_in, t=alphas,
            cond_drop_mask=drop_mask          
        )
        
        losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), seq_distill if self.args.mode == 'distill' else seq, reduction='none')
        losses = losses.mean(-1)

        self.lg('alpha', alphas)
        self.lg('loss', losses)
        self.lg('perplexity', torch.exp(losses.mean())[None].expand(B))
        self.lg('dur', torch.tensor(time.time() - self.last_log_time)[None].expand(B))
        if self.stage == "val":
            cond_in_rounded = torch.round(cond_in * 100) / 100  # 保留两位小数
            # self.lg('condition_input', cond_in_rounded)  # 记录 cond_in
            
            if self.args.mode == 'dirichlet':
                logits_uncond = self.model(
                    xt, torch.zeros_like(cond_full), t=alphas,
                    cond_drop_mask=torch.ones(B, dtype=torch.bool, device=self.device)
                        )
                logits_cond   = self.model(
                    xt, cond_full, t=alphas,
                    cond_drop_mask=torch.zeros(B, dtype=torch.bool, device=self.device)
                )
                logits_pred = logits_uncond + self.args.guidance_scale * (logits_cond - logits_uncond)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            elif self.args.mode == 'fisher':
                logits_pred = self.fisher_inference(seq, cond_full)
                seq_pred = torch.argmax(logits_pred, dim=-1)    
            
            elif self.args.mode == 'riemannian':
                logits_pred = self.riemannian_flow_inference(seq, codon_info)
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
            # self.lg('recovery_uncond', torch.argmax(logits_uncond, -1).eq(seq).float().mean(-1))
            # self.lg('recovery_cond',   torch.argmax(logits_cond,   -1).eq(seq).float().mean(-1))
            # self.lg('guidance_scale', torch.tensor([self.args.guidance_scale]*B, device=self.device))
            
        self.last_log_time = time.time()
        return losses.mean()

    @torch.no_grad()
    def fisher_inference(self, seq, emb):
        B, L = seq.shape
        K = self.model.alphabet_size
        # 初始点：均匀“噪声”
        x = torch.full((B, L, K), 1.0 / np.sqrt(K), device=self.device)

        t_span = torch.linspace(0, 1, self.args.num_integration_steps, device=self.device)
        for s, t in zip(t_span[:-1], t_span[1:]):
            # CFG：无条件/有条件两路向量场
            v_uncond = self.model(
                x, torch.zeros_like(emb), t=s[None].expand(B),
                cond_drop_mask=torch.ones(B, dtype=torch.bool, device=self.device)
            )
            v_cond = self.model(
                x, emb, t=s[None].expand(B),
                cond_drop_mask=torch.zeros(B, dtype=torch.bool, device=self.device)
            )
            v = v_uncond + self.args.guidance_scale * (v_cond - v_uncond)
            v = proj_tangent(x, v)

            # 显式欧拉 + 归一化保持在球面/正正交象限
            x = torch.nn.functional.normalize(x + (t - s) * v, p=2, dim=-1)
            x = torch.clamp(x, min=0)
            x = torch.nn.functional.normalize(x, p=2, dim=-1)

        # 映回 simplex，得到每位点的类别分布，再解码
        p = inv_sphere_map(x)  # (..., K), 每位点概率
        logits_pred = torch.log(p.clamp_min(1e-8))
        return logits_pred

    @torch.no_grad()
    def distill_inference(self, seq, emb):
        B, L = seq.shape
        K = self.model.alphabet_size
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, K, device=seq.device)).sample()
        logits = self.model(x0, emb, t=torch.zeros(B, device=self.device))
        return logits

    @torch.no_grad()
    def riemannian_flow_inference(self, seq, emb, batch_idx=None):
        B, L = seq.shape
        K = self.model.alphabet_size
        xt = torch.distributions.Dirichlet(torch.ones(B, L, K)).sample().to(self.device)
        eye = torch.eye(K).to(self.device)

        t_span = torch.linspace(0, 1, self.args.num_integration_steps, device=self.device)
        for s, t in zip(t_span[:-1], t_span[1:]):
            xt_expanded, prior_weights = expand_simplex(xt, s[None].expand(B), self.args.prior_pseudocount)
            logits = self.model(xt_expanded, emb, s[None].expand(B))
            probs = torch.nn.functional.softmax(logits, -1)
            cond_flows = (eye - xt.unsqueeze(-1)) / (1 - s)
            flow = (probs.unsqueeze(-2) * cond_flows).sum(-1)
            xt = xt + flow * (t - s)

        return logits

    @torch.no_grad()
    def dirichlet_flow_inference(self, seq, emb, model, args):
        """
        x0 sample from initial dirichlet distribution
        eye: identity matrix for calculating conditional flow
        xt
            step_1: xt=x0
            step_n: xt updated in n step
        t_span:
            step_length: 1 to args.alpha_max
            step_#: num_integration_steps
        """
        B, L = seq.shape
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, model.alphabet_size, device=seq.device)).sample()
        eye = torch.eye(model.alphabet_size).to(x0)
        xt = x0

        t_span = torch.linspace(1, args.alpha_max, self.args.num_integration_steps, device=self.device)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])): #here produces xt and xt+1
            prior_weight = args.prior_pseudocount / (s + args.prior_pseudocount - 1)
            seq_xt = torch.cat([xt * (1 - prior_weight), xt * prior_weight], -1)

            logits = model(seq_xt, emb, s[None].expand(B))
            out_probs = torch.nn.functional.softmax(logits / args.flow_temp, -1)

            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)
            if torch.isnan(c_factor).any():
                print(f'NAN cfactor after: xt.min(): {xt.min()}, out_probs.min(): {out_probs.min()}')
                c_factor = torch.nan_to_num(c_factor)

            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (out_probs.unsqueeze(-2) * cond_flows).sum(-1)
            xt = xt + flow * (t - s)
            if not torch.allclose(xt.sum(2), torch.ones((B, L), device=self.device), atol=1e-4) or not (xt >= 0).all():
                print(f'WARNING: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative.')
                xt = simplex_proj(xt)

        return logits, x0

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
        self.distill_model =  PromoterModel(self.distill_args)
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