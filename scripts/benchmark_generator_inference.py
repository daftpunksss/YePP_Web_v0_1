#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import statistics
import time
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictor_web.models.generator_model import DirichletGeneratorModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark legacy vs optimized generator inference.")
    parser.add_argument("--checkpoint", required=True, help="Path to generator checkpoint.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-sequences", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=500)
    parser.add_argument("--condition-dim", type=int, default=84)
    parser.add_argument("--alpha-max", type=float, default=8.0)
    parser.add_argument("--num-integration-steps", type=int, default=120)
    parser.add_argument("--prior-pseudocount", type=float, default=2.0)
    parser.add_argument("--flow-temp", type=float, default=1.0)
    parser.add_argument("--model-guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-values", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--profile", action="store_true", help="Enable per-component profile timings.")
    parser.add_argument(
        "--check-every-n-steps",
        type=int,
        default=10,
        help="Run simplex checks every N steps (when checks enabled).",
    )
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_model(args: argparse.Namespace, device: torch.device) -> DirichletGeneratorModel:
    return DirichletGeneratorModel(
        checkpoint_path=args.checkpoint,
        device=device,
        mode="dirichlet",
        prior_pseudocount=args.prior_pseudocount,
        alpha_max=args.alpha_max,
        num_integration_steps=args.num_integration_steps,
        flow_temp=args.flow_temp,
        guidance_scale=args.model_guidance_scale,
        use_mixed_precision=False,
    )


def run_once(
    model: DirichletGeneratorModel,
    cond: torch.Tensor,
    num_sequences: int,
    seq_length: int,
    guidance_scale: float,
    use_optimized_impl: bool,
    enable_profile: bool,
    check_every_n_steps: int,
) -> tuple[list[str], dict[str, float]]:
    sync_if_cuda(model.device)
    start = time.perf_counter()
    seqs, profile = model.generate_with_profile(
        condition_vectors=cond,
        num_sequences=num_sequences,
        seq_length=seq_length,
        guidance_scale=guidance_scale,
        use_optimized_impl=use_optimized_impl,
        enable_profile=enable_profile,
        enable_step_checks=True,
        check_every_n_steps=check_every_n_steps,
    )
    sync_if_cuda(model.device)
    profile["external_wall_time"] = time.perf_counter() - start
    return seqs, profile


def aggregate(profiles: list[dict[str, float]]) -> dict[str, float]:
    keys = profiles[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        values = [p[key] for p in profiles if isinstance(p[key], (float, int))]
        out[key] = float(statistics.mean(values)) if values else 0.0
    return out


def format_row(mode: str, guidance_scale: float, m: dict[str, float]) -> str:
    steps = max(m.get("num_steps", 1.0), 1.0)
    seqs = max(m.get("num_sequences", 1.0), 1.0)
    seq_per_sec = seqs / max(m.get("sampling_time", 1e-12), 1e-12)
    per_step = m.get("sampling_time", 0.0) / steps
    return (
        f"{mode:10s} | gs={guidance_scale:<4.2f} | total={m.get('external_wall_time', 0.0):.4f}s "
        f"| sampling={m.get('sampling_time', 0.0):.4f}s | seq/s={seq_per_sec:8.2f} | step={per_step:.6f}s"
    )


def print_profile_breakdown(metrics: dict[str, float]) -> None:
    print(
        "    profile: "
        f"forward={metrics.get('model_forward_time', 0.0):.4f}s, "
        f"c_factor={metrics.get('c_factor_time', 0.0):.4f}s, "
        f"checks={metrics.get('checks_time', 0.0):.4f}s, "
        f"update={metrics.get('sampling_update_time', 0.0):.4f}s, "
        f"output={metrics.get('output_conversion_time', 0.0):.4f}s"
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model = make_model(args, device)
    cond = torch.randn(1, args.condition_dim, dtype=torch.float32)

    print("=== Generator Inference Benchmark (same codebase legacy vs optimized) ===")
    print(f"checkpoint={args.checkpoint}")
    print(f"device={device}, warmup={args.warmup}, repeat={args.repeat}, steps={args.num_integration_steps}")

    for guidance_scale in args.guidance_values:
        print(f"\n--- guidance_scale={guidance_scale} ---")
        baseline: list[str] | None = None
        for mode_name, use_optimized in (("legacy", False), ("optimized", True)):
            for i in range(args.warmup):
                set_seed(args.seed + i)
                run_once(
                    model,
                    cond,
                    args.num_sequences,
                    args.seq_length,
                    guidance_scale,
                    use_optimized,
                    enable_profile=args.profile,
                    check_every_n_steps=args.check_every_n_steps,
                )

            profiles: list[dict[str, float]] = []
            final_seqs: list[str] = []
            for i in range(args.repeat):
                set_seed(args.seed + 10_000 + i)
                seqs, prof = run_once(
                    model,
                    cond,
                    args.num_sequences,
                    args.seq_length,
                    guidance_scale,
                    use_optimized,
                    enable_profile=args.profile,
                    check_every_n_steps=args.check_every_n_steps,
                )
                final_seqs = seqs
                profiles.append(prof)

            metrics = aggregate(profiles)
            print(format_row(mode_name, guidance_scale, metrics))
            if args.profile:
                print_profile_breakdown(metrics)

            if baseline is None:
                baseline = final_seqs
            else:
                exact_match = baseline == final_seqs
                print(f"    output_match_vs_legacy={exact_match}")


if __name__ == "__main__":
    main()
