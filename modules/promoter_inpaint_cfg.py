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

from utils.oracle_integrated import uas_infer, cp_infer, specieslm_sc_infer


logger = get_logger(__name__)


class PromoterModule(GeneralModule):
    def __init__(self, args):
        super().__init__(args)

        self.model = PromoterModel(args)
        self.condflow = DirichletConditionalFlow(K=self.model.alphabet_size, alpha_spacing=0.01, alpha_max=args.alpha_max)
        self.loaded_distill_model = False
        self.condition_dropout_prob = 0.1
        
        # args 新增
        self.inpaint_train: bool = True
        self.inpaint_prob: float = 1.0       
        self.inpaint_min_frac: float = 0.05    
        self.inpaint_max_frac: float = 0.95

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
        drop_mask = (torch.rand(B, device=self.device) < self.condition_dropout_prob)
        cond_in = cond_full.clone()
        cond_in[drop_mask] = 0                                       
        
        keep_mask = self._sample_inpaint_keep_mask(B, L, self.device)
        num_supervised = (~keep_mask).sum(dim=1).clamp_min(1)

        if self.args.mode == 'dirichlet' or self.args.mode == 'riemannian':
            xt, alphas = sample_cond_prob_path(self.args, seq, self.model.alphabet_size)
            xt, prior_weights = expand_simplex(xt, alphas, self.args.prior_pseudocount)
            self.lg('prior_weight', prior_weights)
            
            if keep_mask.any():
                one_hot_seq = torch.nn.functional.one_hot(seq, num_classes=self.model.alphabet_size).float()
                xt[keep_mask] = one_hot_seq[keep_mask]
                
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
        
        token_loss = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), 
            seq_distill if self.args.mode == 'distill' else seq, 
            reduction='none')
        
        if self.inpaint_train:
            masked_token_loss = token_loss * (~keep_mask).float()
            losses = masked_token_loss.sum(dim=1) / num_supervised.float()
            ppl_base = masked_token_loss.sum(dim=1) / num_supervised.float()
            self.lg('inpaint_supervised_tokens', num_supervised)
            self.lg('inpaint_keep_frac', keep_mask.float().mean(dim=1))
        else:
            losses = token_loss.mean(dim=1)
            ppl_base = token_loss.mean(dim=1)
            
        #losses = losses.mean(-1)
        #print(f'losses: {losses}')
        self.lg('alpha', alphas)
        self.lg('loss', losses)
        self.lg('perplexity', torch.exp(ppl_base))
        #self.lg('perplexity', torch.exp(losses.mean())[None].expand(B))
        self.lg('dur', torch.tensor(time.time() - self.last_log_time)[None].expand(B))
        if self.stage == "val":
            cond_in_rounded = torch.round(cond_in * 100) / 100  # 保留两位小数
            self.lg('condition_input', cond_in_rounded)  # 记录 cond_in
            
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
            
            rec_mask = (~keep_mask).float()
            denom = rec_mask.sum(dim=1).clamp_min(1.0)
            rec_all = (seq_pred.eq(seq).float() * rec_mask).sum(dim=1) / denom
            
            self.lg('generated_seq', [''.join([['A','C','G','T'][num] for num in seq]) for seq in seq_pred])
            self.lg('ori_seq', [''.join([['A','C','G','T'][num] for num in seq]) for seq in seq])
            #self.lg('recovery', seq_pred.eq(seq).float().mean(-1))
            self.lg('recovery', rec_all)
            self.lg('recovery_uncond', torch.argmax(logits_uncond, -1).eq(seq).float().mean(-1))
            self.lg('recovery_cond',   torch.argmax(logits_cond,   -1).eq(seq).float().mean(-1))
            self.lg('guidance_scale', torch.tensor([self.args.guidance_scale]*B, device=self.device))
            
        self.last_log_time = time.time()
        return losses.mean()

    @torch.no_grad()
    def dirichlet_inpaint_inference(self, cond_seq, keep_mask, emb):
        """
        cond_seq: [B, L] long，已知/锁定位点的碱基索引（未知处值可随意）
        keep_mask: [B, L] bool，True=锁定（已知）
        emb: 条件输入，与训练相同（codon_info+species 的拼接）
        返回: logits, seq_pred
        """
        B, L = cond_seq.shape
        K = self.model.alphabet_size
        device = cond_seq.device

        # 初始 x0 ~ Dirichlet(1)
        xt = torch.distributions.Dirichlet(torch.ones(B, L, K, device=device)).sample()
        eye = torch.eye(K, device=device)
        one_hot_cond = torch.nn.functional.one_hot(cond_seq, num_classes=K).float()

        # 把锁定位点强制为 one-hot
        if keep_mask.any():
            xt[keep_mask] = one_hot_cond[keep_mask]

        t_span = torch.linspace(1, self.args.alpha_max, self.args.num_integration_steps, device=device)
        for s, t in zip(t_span[:-1], t_span[1:]):
            prior_weight = self.args.prior_pseudocount / (s + self.args.prior_pseudocount - 1)
            seq_xt = torch.cat([xt * (1 - prior_weight), xt * prior_weight], -1)

            logits = self.model(seq_xt, emb, s[None].expand(B))
            out_probs = torch.nn.functional.softmax(logits / self.args.flow_temp, -1)

            # cond flow（DFM 标准形式）
            c_factor = self.condflow.c_factor(xt.detach().cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)
            c_factor = torch.nan_to_num(c_factor)
            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)  # [B, L, K, K]

            # 关键：锁定位点零流
            if keep_mask.any():
                cond_flows[keep_mask] = 0.0

            flow = (out_probs.unsqueeze(-2) * cond_flows).sum(-1)          # [B, L, K]
            xt = xt + flow * (t - s)

            # 数值误差纠正 + 再次钉住锁定位点
            xt = simplex_proj(xt)
            if keep_mask.any():
                xt[keep_mask] = one_hot_cond[keep_mask]

        seq_pred = torch.argmax(xt, dim=-1)
        return logits, seq_pred

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
                
    # 放到 PromoterModule 类里（建议在 lg() 之上）
    def _sample_inpaint_keep_mask(self, B, L, device):
        if not getattr(self.args, 'inpaint_train', False):
            return torch.zeros(B, L, dtype=torch.bool, device=device)

        trigger = torch.rand(B, device=device) < getattr(self.args, 'inpaint_prob', 0.5)
        keep_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        if not trigger.any():
            return keep_mask

        min_frac = float(getattr(self.args, 'inpaint_min_frac', 0.15))
        max_frac = float(getattr(self.args, 'inpaint_max_frac', 0.50))
        max_frac = max(min(max_frac, 1.0), 0.0)
        min_frac = max(min(min_frac, max_frac), 0.0)

        for b in torch.nonzero(trigger, as_tuple=False).flatten().tolist():
            frac = min_frac + (max_frac - min_frac) * torch.rand(1, device=device).item()
            k = max(1, int(round(frac * L)))
            idx = torch.randperm(L, device=device)[:k]
            keep_mask[b, idx] = True
        return keep_mask


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