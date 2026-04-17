from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
from functools import lru_cache
import time
from typing import Any
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

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _profile_pack(self) -> dict[str, Any]:
        return {
            "total_wall_time": 0.0,
            "sampling_time": 0.0,
            "model_forward_time": 0.0,
            "c_factor_time": 0.0,
            "checks_time": 0.0,
            "sampling_update_time": 0.0,
            "output_conversion_time": 0.0,
            "num_steps": 0.0,
            "num_sequences": 0.0,
            "seq_length": 0.0,
            "mode": "",
            "guidance_scale": 0.0,
        }

    @torch.inference_mode()
    def generate(
        self,
        condition_vectors: torch.Tensor,
        num_sequences: int,
        seq_length: int,
        guidance_scale: float | None = None,
        use_optimized_impl: bool = True,
        enable_profile: bool = False,
        enable_step_checks: bool = True,
        check_every_n_steps: int = 10,
    ) -> list[str]:
        sequences, _ = self.generate_with_profile(
            condition_vectors=condition_vectors,
            num_sequences=num_sequences,
            seq_length=seq_length,
            guidance_scale=guidance_scale,
            use_optimized_impl=use_optimized_impl,
            enable_profile=enable_profile,
            enable_step_checks=enable_step_checks,
            check_every_n_steps=check_every_n_steps,
        )
        return sequences

    @torch.inference_mode()
    def generate_with_profile(
        self,
        condition_vectors: torch.Tensor,
        num_sequences: int,
        seq_length: int,
        guidance_scale: float | None = None,
        use_optimized_impl: bool = True,
        enable_profile: bool = False,
        enable_step_checks: bool = True,
        check_every_n_steps: int = 10,
    ) -> tuple[list[str], dict[str, Any]]:
        if condition_vectors.ndim != 2:
            raise ValueError("Condition vectors must be rank-2 with shape [N, cond_dim].")

        if num_sequences <= 0:
            raise ValueError("num_sequences must be > 0.")
        if check_every_n_steps <= 0:
            raise ValueError("check_every_n_steps must be > 0.")

        profile = self._profile_pack()
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
        zeros_cond_drop = torch.cat([cond_drop_true, cond_drop_false], dim=0)
        c_factor_recorder = None
        if enable_profile:
            def c_factor_recorder(duration: float) -> None:
                profile["c_factor_time"] += duration

        simplex_target = torch.ones((num_sequences, seq_length), device=self.device)
        total_start = time.perf_counter()
        sampling_start = time.perf_counter()
        for step_idx, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            prior_weight = self.args.prior_pseudocount / (s + self.args.prior_pseudocount - 1)
            seq_xt = torch.cat([xt * (1 - prior_weight), xt * prior_weight], dim=-1)
            t_tensor = s[None].expand(num_sequences)

            with self._autocast_context():
                if enable_profile:
                    self._sync_device()
                    forward_start = time.perf_counter()
                if gs != 0:
                    if use_optimized_impl:
                        seq_xt_2b = torch.cat([seq_xt, seq_xt], dim=0)
                        cond_2b = torch.cat([zeros_cond, cond_full], dim=0)
                        t_tensor_2b = s[None].expand(2 * num_sequences)
                        logits_2b = self.model(
                            seq_xt_2b,
                            cond_2b,
                            t=t_tensor_2b,
                            cond_drop_mask=zeros_cond_drop,
                        )
                        logits_uncond, logits_cond = logits_2b.chunk(2, dim=0)
                    else:
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
                if enable_profile:
                    self._sync_device()
                    profile["model_forward_time"] += time.perf_counter() - forward_start

            c_factor = self.condflow.c_factor(
                xt.detach().cpu().numpy(),
                s.item(),
                profile_recorder=c_factor_recorder,
            )
            c_factor = torch.from_numpy(c_factor).to(xt)
            c_factor = torch.nan_to_num(c_factor)

            if enable_profile:
                self._sync_device()
                update_start = time.perf_counter()
            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (out_probs.unsqueeze(-2) * cond_flows).sum(dim=-1)
            xt = xt + flow * (t - s)
            if enable_profile:
                self._sync_device()
                profile["sampling_update_time"] += time.perf_counter() - update_start

            should_check = enable_step_checks and (step_idx % check_every_n_steps == 0)
            if should_check:
                if enable_profile:
                    self._sync_device()
                    check_start = time.perf_counter()
                if not torch.allclose(xt.sum(2), simplex_target, atol=1e-4) or not (xt >= 0).all():
                    xt = simplex_proj(xt)
                if enable_profile:
                    self._sync_device()
                    profile["checks_time"] += time.perf_counter() - check_start

        self._sync_device()
        profile["sampling_time"] = time.perf_counter() - sampling_start
        output_start = time.perf_counter()
        generated_seq = torch.argmax(xt, dim=-1)
        sequences = ["".join(ALPHABET[idx] for idx in seq.tolist()) for seq in generated_seq]
        self._sync_device()
        profile["output_conversion_time"] = time.perf_counter() - output_start
        profile["total_wall_time"] = time.perf_counter() - total_start
        profile["num_steps"] = float(max(0, self.args.num_integration_steps - 1))
        profile["num_sequences"] = float(num_sequences)
        profile["seq_length"] = float(seq_length)
        profile["mode"] = "optimized" if use_optimized_impl else "legacy"
        profile["guidance_scale"] = float(gs)
        return sequences, profile


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
