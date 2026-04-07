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
#from model.t_emb_seed import PromoterModel

from utils.oracle_integrated import uas_infer, cp_infer, specieslm_sc_infer


logger = get_logger(__name__)


class PromoterModule(GeneralModule):
    def __init__(self, args):
        super().__init__(args)

        self.model = PromoterModel(args)
        self.condflow = DirichletConditionalFlow(K=self.model.alphabet_size, alpha_spacing=0.01, alpha_max=args.alpha_max)
        self.loaded_distill_model = False
        
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
        seq = torch.argmax(seq_one_hot, dim=-1)
        #print(f'seq_shape:{seq.shape}')
        B, L = seq.shape

        if self.args.mode == 'dirichlet' or self.args.mode == 'riemannian':
            xt, alphas = sample_cond_prob_path(self.args, seq, self.model.alphabet_size)
            xt, prior_weights = expand_simplex(xt, alphas, self.args.prior_pseudocount)
            self.lg('prior_weight', prior_weights)
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
        logits = self.model(xt, codon_info, t=alphas)
        
        losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), seq_distill if self.args.mode == 'distill' else seq, reduction='none')
        losses = losses.mean(-1)

        self.lg('alpha', alphas)
        self.lg('loss', losses)
        self.lg('perplexity', torch.exp(losses.mean())[None].expand(B))
        self.lg('dur', torch.tensor(time.time() - self.last_log_time)[None].expand(B))
        if self.stage == "val":

            if self.args.mode == 'dirichlet':
                logits_pred, _ = self.dirichlet_flow_inference(seq, codon_info, self.model, args=self.args)
                # print(f'logits_pred: {logits_pred.shape}')
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
            
            # generated_seq = [''.join([['A','C','G','T'][num] for num in seq]) for seq in seq_pred]
            # ori_seq = [''.join([['A','C','G','T'][num] for num in seq]) for seq in seq]
            
            self.lg('generated_seq', [''.join([['A','C','G','T'][num] for num in seq]) for seq in seq_pred])
            self.lg('ori_seq', [''.join([['A','C','G','T'][num] for num in seq]) for seq in seq])
            # seq_pred_one_hot = torch.nn.functional.one_hot(seq_pred, num_classes=self.model.alphabet_size).float()
            #print(f'seq_input_ont_hot shape: {seq_one_hot.shape}')
            # print(f'seq_pred_ont_hot shape: {seq_pred_one_hot.shape}')
            # uas_input_strength = self.uas_oracle(seq_one_hot.float())
            # cp_input_strength = self.cp_oracle(seq_one_hot[:, -119:, :].float())
            
            # uas_pred_strength = self.uas_oracle(seq_pred_one_hot)
            # cp_pred_strength = self.cp_oracle(seq_pred_one_hot[:, -119:, :].float())
            # print(f'cp_pred_strength:{cp_pred_strength.shape}')
            # print(f'cp_input_strength:{cp_input_strength.shape}')
            self.lg('recovery', seq_pred.eq(seq).float().mean(-1))
            
            # ori_strength = specieslm_sc_infer(ori_seq).detach().cpu().numpy()
            # grd_strength = specieslm_sc_infer(generated_seq).detach().cpu().numpy()
            
            # uas_input_strength_np = uas_input_strength.detach().cpu().numpy()
            # cp_input_strength_np = cp_input_strength.detach().cpu().numpy()
            # uas_pred_strength_np = uas_pred_strength.detach().cpu().numpy()
            # cp_pred_strength_np = cp_pred_strength.detach().cpu().numpy()

            # self.lg('input_seq_pred_strength', ori_strength)
            # self.lg('generated_seq_pred_strength', grd_strength)
            # self.lg('uas_generated_pred_strength', uas_pred_strength_np)
            # self.lg('cp_generated_pred_strength', cp_pred_strength_np)

            # self.lg('strength_difference', (ori_strength - grd_strength).tolist())
            # self.lg('cp_strength_difference', (cp_pred_strength_np - cp_input_strength_np).tolist())


        self.last_log_time = time.time()
        return losses.mean()
    
    def atcg_freq(self, seqs):
        """
        计算 N 轮采样后，每个位置的 A/T/C/G 频率分布
        输入：
            seqs: (N, B, L) - 生成的 N 轮序列
        输出：
            freqs: (B, L, 4) - 每个 batch 内，每个位置的 A/T/C/G 频率
        """
        B, L = seqs.shape[1], seqs.shape[2]
        freqs = torch.zeros((B, L, 4), device=seqs.device)

        for i in range(4):  # A=0, T=1, C=2, G=3
            freqs[:, :, i] = (seqs == i).mean(dim=0)  # 计算每个位置的频率

        return freqs


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

