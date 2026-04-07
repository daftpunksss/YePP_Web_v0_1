import copy

import numpy as np
import pandas as pd
import torch, time, os
import swanlab
import yaml
import torchmetrics
from torchmetrics import metric

from torch import optim
from torch import nn

from utils.esm import upgrade_state_dict
from modules.general_module import GeneralModule
from utils.logging import get_logger
from model.legnet import OptimizedLegNet
#from model.original_legnet import LegNet
from model.ffn import FFN
from model.ShallowConv import shallowconv
from model.nt_linear import RegressionModelFreeze
from model.unet import UNET1DSegmentationHead, UNET1DRegressor, UNET1DDistribution
from model.finetune_unet import SegmentNT
from model.finetune_nt2B import NT2B
from model.nt_linear import RegressionModelFreeze
from model.hybrid import HybridNet, SCHybridNet, CssHybridNet, DeepExpressionNet, ResNTEmbeddingNet, TENet, FFNet
from model.finetune_lastlayer_llm import NTRegressor
from model.utr import UTRNet
from model.spcieslm_ft import SpeciesLMLinearRegressor, SpeciesLMLightAttention
from utils.ordinal_regression_loss import CornLoss


logger = get_logger(__name__)


class OracleModule(GeneralModule):
    def __init__(self, args):
        super().__init__(args)

        self.model = SpeciesLMLightAttention()
        self.using_mse = False
        self.using_cls = False
        self.using_kldiv = False

        if args.mse:
            self.using_mse = True
            self.loss = torch.nn.MSELoss()
            print('***************************************')
            print('using mse loss......')
            self.pearson_corr = torchmetrics.PearsonCorrCoef(num_outputs=1)
            self.spearman_corr = torchmetrics.SpearmanCorrCoef(num_outputs=1)
            self.r2 = torchmetrics.R2Score(num_outputs=1, multioutput="raw_values")
            self.all_val_pred_scores = []
            self.all_val_y_scores = []

        if args.cls:
            self.using_cls = True
            self.loss = torch.nn.CrossEntropyLoss()
            #self.loss = CornLoss(num_classes=20)
            print('***************************************')
            print('using cls loss......')
            self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=20)
            self.prec = torchmetrics.Precision(task='multiclass', average='macro', num_classes=20)
            self.f1_score = torchmetrics.F1Score(task='multiclass', average='macro', num_classes=20)
            self.recall = torchmetrics.Recall(task='multiclass', average='macro', num_classes=20)
            self.mcc = torchmetrics.MatthewsCorrCoef(task='multiclass', num_classes=20)
            self.all_val_pred_scores = []
            self.all_val_y_scores = []
            
        if args.kldiv:
            self.using_kldiv = True
            print("***************************************\nusing KL divergence loss......")
            # KLDivLoss 期望输入 = log_prob，target = prob；batchmean = 1/B * sum(KL_i)
            self.loss = nn.KLDivLoss(reduction="batchmean", log_target=False)

            # 用“质心坐标”评估 → 仍是标量，因此 num_outputs=1
            self.pearson_corr  = torchmetrics.PearsonCorrCoef(num_outputs=1)
            self.spearman_corr = torchmetrics.SpearmanCorrCoef(num_outputs=1)
            self.r2            = torchmetrics.R2Score(num_outputs=1, multioutput="raw_values")

            # 累积结果
            self.all_val_pred_scores = []   # 质心
            self.all_val_y_scores    = []

            # 方便 later 求质心：register_buffer 保证 device 正确
            self.register_buffer("coord_idx", torch.arange(args.seq_len, dtype=torch.float32))

            

    def on_load_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {k: v for k,v in checkpoint['state_dict'].items()}

    def training_step(self, batch, batch_idx):
        self.stage = 'train'
        loss = self.general_step(batch, batch_idx)
        if self.args.ckpt_iterations is not None and self.trainer.global_step in self.args.ckpt_iterations:
            self.trainer.save_checkpoint(os.path.join(os.environ["MODEL_DIR"],f"epoch={self.trainer.current_epoch}-step={self.trainer.global_step}.ckpt"))
        return loss

    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        if self.args.validate:
            self.try_print_log()

    def general_step(self, batch, batch_idx=None):
        self.iter_step += 1
        
        #print('**************************************')
        #print(f'y_probs.shape:{y_probs.shape},y_score.shape:{y_score.shape}')
        #print(f'y_score:{y_score}')
        #print('**************************************')

        if self.using_mse:
            #seqs, tpm = batch
            inputs_id, masks_id, tpm = batch['input_ids'], batch['attention_mask'], batch['labels']
            #print(f'y_cls:{y_cls}')
            #print(f'y_score dtype:{y_score.dtype}')
            #print(f'seqs type: {seqs.dtype}, yscore type: {y_score.dtype}')
            #y_score = y_score.squeeze(-1)
            preds = self.model(inputs_id, masks_id).squeeze()
            #print(pred_score.dtype)
            #print('**************************************')
            #print(f'pred_score:{pred_score.shape}, y_score.shape:{y_score.shape}')
            #print(f'y_score:{y_score}')
            #print('**************************************')
            losses = self.loss(preds, tpm)
            losses = losses.mean(-1)
            B = len(tpm)
            # B, _, _ = seqs.shape
            # B, _, = seqs.shape
        if self.using_cls:
            seqs, y_cls = batch
            #print(f'seqs dtype:{seqs.dtype}')
            #print(f'y_score dtype:{y_score.dtype}')
            #print(f'seqs type: {seqs.dtype}, yscore type: {y_score.dtype}')
            #y_score = y_score.squeeze(-1)
            pred_cls = self.model(seqs)
            #print(pred_score.dtype)
            #print('**************************************')
            #print(f'pred_score:{pred_score.shape}, y_score.shape:{y_score.shape}')
            #print(f'y_score:{y_score}')
            #print('**************************************')
            losses = self.loss(pred_cls, y_cls)
            losses = losses.mean(-1)
            B,_,_ = seqs.shape
            
        if self.using_kldiv:
            seqs, p_target = batch                 # p_target shape [B,L], 已归一化
            logits   = self.model(seqs)            # [B,L]
            log_prob = torch.log_softmax(logits, dim=-1)

            losses = self.loss(log_prob, p_target)
            losses = losses.mean(-1)
            B = seqs.size(0)
    

        # else:
        #     seqs, y_probs, y_score = batch
        #     y_score = y_score.squeeze(-1)
        #     logprobs, pred_score = self.model(seqs)
        #     losses = self.loss(logprobs, y_probs)
        #     losses = losses.mean(-1)
        #     B, _, _ = seqs.shape
        #print(f'tpm:{y_probs}, pred_tpm:{logprobs}')
        
        #print(losses)
        self.lg('loss', losses.expand(B))
        self.lg('dur', torch.tensor(time.time() - self.last_log_time)[None].expand(B))
        if self.stage == "val":
            with torch.no_grad():
                if self.using_mse:
                    preds = self.model(inputs_id, masks_id).squeeze()

                    self.all_val_pred_scores.append(preds.detach())
                    self.all_val_y_scores.append(tpm.detach())
                    
                    # pearson_corr = self.pearson_corr(pred_score, y_score)
                    # spearman_corr = self.spearman_corr(pred_score, y_score)
                    # r2 = self.r2(pred_score, y_score)
                    # self.lg('pearson', pearson_corr.clone().detach()[None].expand(B))
                    # self.lg('spearman', spearman_corr.clone().detach()[None].expand(B))
                    # self.lg('R2', r2.clone().detach()[None].expand(B))
                    self.lg('y_socres', tpm)
                    self.lg('pred_socres', preds)
                if self.using_cls:
                    pred_cls = self.model(seqs)

                    self.all_val_pred_scores.append(pred_cls.detach())
                    self.all_val_y_scores.append(y_cls.detach())
                    
                if self.using_kldiv:
                    prob = torch.softmax(logits, dim=-1)
                    pred_center  = (prob * self.coord_idx).sum(-1, keepdim=True)
                    true_center  = (p_target * self.coord_idx).sum(-1, keepdim=True)
                    self.all_val_pred_scores.append(pred_center.detach())
                    self.all_val_y_scores.append(true_center.detach())

                # else:
                #     logprobs, pred_score = self.model(seqs)
                #     self.all_pred_scores.append(pred_score.detach().cpu())
                #     self.all_y_scores.append(y_score.detach().cpu())
                #     #print('*************************************')
                #     #print(f'logprob:{logprobs.shape},pred_score:{pred_score},y_prob:{y_probs.shape}')
                #     #print('*************************************')
                #     pearson_corr = self.pearson_corr(pred_score, y_score)
                #     spearman_corr = self.spearman_corr(pred_score, y_score)
                #     r2 = self.r2(pred_score, y_score)
                #     self.lg('pearson', pearson_corr.clone().detach()[None].expand(B))
                #     self.lg('spearman', spearman_corr.clone().detach()[None].expand(B))
                #     self.lg('R2', r2.clone().detach()[None].expand(B))
                #     #self.lg('msur_logprobs', [y_prob for y_prob in y_probs])
                #     #self.lg('pred_logprobs', [logprob for logprob in logprobs])
                #     self.lg('y_socres', y_score)
                #     self.lg('pred_socres', pred_score)
                    
        self.last_log_time = time.time()
        return losses.mean()

    @torch.no_grad()
    def on_validation_epoch_start(self) -> None:
        self.generator = np.random.default_rng(seed=137)

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        
        all_pred_scores = torch.cat(self.all_val_pred_scores, dim=0).squeeze()
        all_y_scores = torch.cat(self.all_val_y_scores, dim=0)

        if self.using_mse:
            pearson_corr = self.pearson_corr(all_pred_scores, all_y_scores)
            spearman_corr = self.spearman_corr(all_pred_scores, all_y_scores)
            r2 = self.r2(all_pred_scores, all_y_scores)

            B = all_pred_scores.shape[0]
            # self.lg('pearson_TPM', pearson_corr[0].clone().detach()[None].expand(B))
            # self.lg('spearman_TPM', spearman_corr[0].clone().detach()[None].expand(B))
            # self.lg('R2_TPM', r2[0].clone().detach()[None].expand(B))
            self.lg('pearson', pearson_corr.clone().detach()[None].expand(B))
            self.lg('spearman', spearman_corr.clone().detach()[None].expand(B))
            self.lg('R2', r2.clone().detach()[None].expand(B))
            # self.lg('pearson_PSS', pearson_corr[2].clone().detach()[None].expand(B))
            # self.lg('spearman_PSS', spearman_corr[2].clone().detach()[None].expand(B))
            # self.lg('R2_PSS', r2[2].clone().detach()[None].expand(B))
            
            # metrics_to_log = {
            #         "val_pcc_TPM":     pearson_corr[0],
            #         "val_pcc_UTRlen":  pearson_corr[1],
            #         "val_pcc_PSS":     pearson_corr[2],
            #         "val_spc_TPM":     spearman_corr[0],
            #         "val_spc_UTRlen":     spearman_corr[1],
            #         "val_spc_PSS":     spearman_corr[2],
            #         "val_r2_TPM":     r2[0],
            #         "val_r2_UTRlen":     r2[1],
            #         "val_r2_PSS":     r2[2],
            #     }
            # self.log_dict(metrics_to_log, prog_bar=True, sync_dist=True)
        if self.using_cls:
            acc = self.acc(all_pred_scores, all_y_scores)
            prec = self.prec(all_pred_scores, all_y_scores)
            f1_score = self.f1_score(all_pred_scores, all_y_scores)
            recall = self.recall(all_pred_scores, all_y_scores)
            mcc = self.mcc(all_pred_scores, all_y_scores)

            B = all_pred_scores.shape[0]
            self.lg('acc', acc.clone().detach()[None].expand(B))
            self.lg('prec', prec.clone().detach()[None].expand(B))
            self.lg('f1_score', f1_score.clone().detach()[None].expand(B))
            self.lg('recall', recall.clone().detach()[None].expand(B))
            self.lg('mcc', mcc.clone().detach()[None].expand(B))
            
        if self.using_kldiv:
            all_pred_scores = torch.cat(self.all_val_pred_scores, dim=0)  # [N,1]
            all_y_scores    = torch.cat(self.all_val_y_scores,  dim=0)

            pearson_corr  = self.pearson_corr(all_pred_scores, all_y_scores)
            spearman_corr = self.spearman_corr(all_pred_scores, all_y_scores)
            r2            = self.r2(all_pred_scores, all_y_scores)

            B = all_pred_scores.shape[0]
            self.lg('pearson_center',  pearson_corr.clone().detach()[None].expand(B))
            self.lg('spearman_center', spearman_corr.clone().detach()[None].expand(B))
            self.lg('R2_center',       r2.clone().detach()[None].expand(B))


        self.all_val_pred_scores = []
        self.all_val_y_scores = []

        
        self.generator = np.random.default_rng()
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
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
    
    
    
