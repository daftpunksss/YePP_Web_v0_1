#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictor_web.models.generator_model import DirichletGeneratorModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal regression: legacy vs optimized equivalence.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-sequences", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--condition-dim", type=int, default=84)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--num-integration-steps", type=int, default=40)
    parser.add_argument("--alpha-max", type=float, default=8.0)
    parser.add_argument("--flow-temp", type=float, default=1.0)
    parser.add_argument("--prior-pseudocount", type=float, default=2.0)
    parser.add_argument("--check-every-n-steps", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(args: argparse.Namespace, device: torch.device) -> DirichletGeneratorModel:
    return DirichletGeneratorModel(
        checkpoint_path=args.checkpoint,
        device=device,
        mode="dirichlet",
        prior_pseudocount=args.prior_pseudocount,
        alpha_max=args.alpha_max,
        num_integration_steps=args.num_integration_steps,
        flow_temp=args.flow_temp,
        guidance_scale=args.guidance_scale,
        use_mixed_precision=False,
    )


def compare_sequences(legacy: list[str], optimized: list[str]) -> tuple[bool, float]:
    if len(legacy) != len(optimized):
        return False, 0.0
    total = 0
    match = 0
    for lseq, oseq in zip(legacy, optimized):
        if len(lseq) != len(oseq):
            return False, 0.0
        for lc, oc in zip(lseq, oseq):
            total += 1
            if lc == oc:
                match += 1
    ratio = (match / total) if total else 1.0
    return legacy == optimized, ratio


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = make_model(args, device)
    cond = torch.randn(1, args.condition_dim, dtype=torch.float32)

    set_seed(args.seed)
    legacy, _ = model.generate_with_profile(
        condition_vectors=cond,
        num_sequences=args.num_sequences,
        seq_length=args.seq_length,
        guidance_scale=args.guidance_scale,
        use_optimized_impl=False,
        enable_profile=False,
        enable_step_checks=True,
        check_every_n_steps=args.check_every_n_steps,
    )

    set_seed(args.seed)
    optimized, _ = model.generate_with_profile(
        condition_vectors=cond,
        num_sequences=args.num_sequences,
        seq_length=args.seq_length,
        guidance_scale=args.guidance_scale,
        use_optimized_impl=True,
        enable_profile=False,
        enable_step_checks=True,
        check_every_n_steps=args.check_every_n_steps,
    )

    strict_equal, token_match_ratio = compare_sequences(legacy, optimized)
    print(f"strict_equal={strict_equal}")
    print(f"token_match_ratio={token_match_ratio:.6f}")
    if not strict_equal:
        raise SystemExit("Regression failed: optimized output differs from legacy output.")


if __name__ == "__main__":
    main()
