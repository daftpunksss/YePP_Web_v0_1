from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from predictor_web import config
from predictor_web.models.generator_model import get_generator_model
from predictor_web.services.cfg_condition_service import get_cfg_condition_resolver
from predictor_web.utils.generator_io import (
    build_generation_dataframe,
    write_generation_csv,
    write_generation_fasta,
)


def _resolve_device() -> torch.device:
    if config.GENERATOR_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(config.GENERATOR_DEVICE)


def _parse_condition_vector(text: str, expected_dim: int) -> torch.Tensor:
    cleaned = (text or "").strip()
    if not cleaned:
        return torch.zeros(1, expected_dim, dtype=torch.float32)

    values = [chunk.strip() for chunk in cleaned.split(",") if chunk.strip()]
    if len(values) != expected_dim:
        raise ValueError(f"Condition vector must contain exactly {expected_dim} comma-separated floats.")

    try:
        parsed = [float(v) for v in values]
    except ValueError as err:
        raise ValueError("Condition vector contains non-numeric values.") from err

    return torch.tensor([parsed], dtype=torch.float32)


def run_generation(
    checkpoint_key: str,
    num_sequences: int,
    guidance_scale: float,
    selected_species: str,
    gene_query: str,
    manual_condition_vector_csv: str,
) -> tuple[pd.DataFrame, str, str, dict[str, str]]:
    if checkpoint_key not in config.GENERATOR_CHECKPOINTS:
        raise ValueError("Unsupported generator checkpoint selected.")

    if num_sequences < 1 or num_sequences > config.GENERATOR_MAX_SEQUENCES:
        raise ValueError(
            f"num_sequences must be between 1 and {config.GENERATOR_MAX_SEQUENCES}."
        )

    device = _resolve_device()
    checkpoint_path = config.GENERATOR_CHECKPOINTS[checkpoint_key]

    model = get_generator_model(
        checkpoint_path=str(checkpoint_path),
        device_str=str(device),
        mode=config.GENERATOR_MODE,
        prior_pseudocount=config.GENERATOR_PRIOR_PSEUDOCOUNT,
        alpha_max=config.GENERATOR_ALPHA_MAX,
        num_integration_steps=config.GENERATOR_NUM_INTEGRATION_STEPS,
        flow_temp=config.GENERATOR_FLOW_TEMP,
        guidance_scale=config.GENERATOR_GUIDANCE_SCALE,
        use_mixed_precision=config.GENERATOR_USE_MIXED_PRECISION,
    )

    resolver = get_cfg_condition_resolver()
    manual_override = (manual_condition_vector_csv or "").strip()
    if manual_override:
        cond = _parse_condition_vector(
            manual_condition_vector_csv, expected_dim=config.GENERATOR_CONDITION_DIM
        )
        summary = {
            "selected_species": selected_species,
            "requested_gene": gene_query,
            "matched_gene_id": "manual_override",
            "condition_resolved": "yes (manual override)",
            "condition_dim": str(cond.shape[1]),
        }
    else:
        resolved = resolver.resolve(
            selected_species=selected_species,
            gene_query=gene_query,
            expected_condition_dim=config.GENERATOR_CONDITION_DIM,
        )
        cond = resolved.condition
        summary = {
            "selected_species": resolved.selected_species,
            "requested_gene": resolved.requested_gene,
            "matched_gene_id": resolved.matched_gene_id,
            "condition_resolved": "yes",
            "condition_dim": str(cond.shape[1]),
        }

    sequences = model.generate(
        condition_vectors=cond,
        num_sequences=num_sequences,
        seq_length=config.GENERATOR_SEQUENCE_LENGTH,
        guidance_scale=guidance_scale,
    )

    df = build_generation_dataframe(sequences, checkpoint_name=checkpoint_key)
    run_id = datetime.utcnow().strftime("generator_%Y%m%d_%H%M%S")
    csv_path = write_generation_csv(df, config.GENERATOR_OUTPUT_DIR, run_id)
    fasta_path = write_generation_fasta(df, config.GENERATOR_OUTPUT_DIR, run_id)
    return df, csv_path, fasta_path, summary
