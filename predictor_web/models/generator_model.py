from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
from functools import lru_cache
import torch
from torch import nn

from model.promoter_model import PromoterModel
from utils.flow_utils import DirichletConditionalFlow, simplex_proj


ALPHABET = ("A", "C", "G", "T")


class DirichletGeneratorModel:
    """Inference-only wrapper for Dirichlet flow matching promoter generation."""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        mode: str,
        prior_pseudocount: float,
        alpha_max: float,
        num_integration_steps: int,
        flow_temp: float,
        guidance_scale: float,
        use_mixed_precision: bool,
    ) -> None:
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device.type == "cuda"
        self.args = Namespace(
            mode=mode,
            prior_pseudocount=prior_pseudocount,
            alpha_max=alpha_max,
            num_integration_steps=num_integration_steps,
            flow_temp=flow_temp,
            guidance_scale=guidance_scale,
        )
        self.model = self._load_model(checkpoint_path).eval()
        self.condflow = DirichletConditionalFlow(
            K=self.model.alphabet_size,
            alpha_spacing=0.01,
            alpha_max=self.args.alpha_max,
        )

    def _load_model(self, checkpoint_path: str) -> nn.Module:
        model = PromoterModel(self.args).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "state_dict" in checkpoint:
            state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)
        return model

    def _autocast_context(self):
        if not self.use_mixed_precision:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    @torch.inference_mode()
    def generate(
        self,
        condition_vectors: torch.Tensor,
        num_sequences: int,
        seq_length: int,
        guidance_scale: float | None = None,
    ) -> list[str]:
        if condition_vectors.ndim != 2:
            raise ValueError("Condition vectors must be rank-2 with shape [N, cond_dim].")

        if num_sequences <= 0:
            raise ValueError("num_sequences must be > 0.")

        cond_count = condition_vectors.shape[0]
        repeats = (num_sequences + cond_count - 1) // cond_count
        cond_full = condition_vectors.to(self.device).repeat(repeats, 1)[:num_sequences]

        xt = torch.distributions.Dirichlet(
            torch.ones(num_sequences, seq_length, len(ALPHABET), device=self.device)
        ).sample()
        eye = torch.eye(len(ALPHABET), device=self.device)

        t_span = torch.linspace(
            1,
            self.args.alpha_max,
            self.args.num_integration_steps,
            device=self.device,
        )

        gs = self.args.guidance_scale if guidance_scale is None else guidance_scale
        cond_drop_true = torch.ones(num_sequences, dtype=torch.bool, device=self.device)
        cond_drop_false = torch.zeros(num_sequences, dtype=torch.bool, device=self.device)
        zeros_cond = torch.zeros_like(cond_full)

        simplex_target = torch.ones((num_sequences, seq_length), device=self.device)
        for s, t in zip(t_span[:-1], t_span[1:]):
            prior_weight = self.args.prior_pseudocount / (s + self.args.prior_pseudocount - 1)
            seq_xt = torch.cat([xt * (1 - prior_weight), xt * prior_weight], dim=-1)
            t_tensor = s[None].expand(num_sequences)

            with self._autocast_context():
                if gs != 0:
                    logits_uncond = self.model(
                        seq_xt,
                        zeros_cond,
                        t=t_tensor,
                        cond_drop_mask=cond_drop_true,
                    )
                    logits_cond = self.model(
                        seq_xt,
                        cond_full,
                        t=t_tensor,
                        cond_drop_mask=cond_drop_false,
                    )
                    logits = logits_uncond + gs * (logits_cond - logits_uncond)
                else:
                    logits = self.model(
                        seq_xt,
                        zeros_cond,
                        t=t_tensor,
                        cond_drop_mask=cond_drop_true,
                    )

                out_probs = torch.nn.functional.softmax(logits / self.args.flow_temp, dim=-1)

            c_factor = self.condflow.c_factor(xt.detach().cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)
            c_factor = torch.nan_to_num(c_factor)

            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (out_probs.unsqueeze(-2) * cond_flows).sum(dim=-1)
            xt = xt + flow * (t - s)

            if not torch.allclose(xt.sum(2), simplex_target, atol=1e-4) or not (xt >= 0).all():
                xt = simplex_proj(xt)

        generated_seq = torch.argmax(xt, dim=-1)
        return ["".join(ALPHABET[idx] for idx in seq.tolist()) for seq in generated_seq]


@lru_cache(maxsize=8)
def get_generator_model(
    checkpoint_path: str,
    device_str: str,
    mode: str,
    prior_pseudocount: float,
    alpha_max: float,
    num_integration_steps: int,
    flow_temp: float,
    guidance_scale: float,
    use_mixed_precision: bool,
) -> DirichletGeneratorModel:
    device = torch.device(device_str)
    return DirichletGeneratorModel(
        checkpoint_path=checkpoint_path,
        device=device,
        mode=mode,
        prior_pseudocount=prior_pseudocount,
        alpha_max=alpha_max,
        num_integration_steps=num_integration_steps,
        flow_temp=flow_temp,
        guidance_scale=guidance_scale,
        use_mixed_precision=use_mixed_precision,
    )
