from __future__ import annotations

import os
import time

from predictor_web.services.generator_service import run_generation
from predictor_web.services.predict_service import score_sequences_for_models


DEFAULT_GENERATOR_SCORING_MODELS = ["sc", "pp"]


def run_generation_with_optional_scoring(
    checkpoint_key: str,
    num_sequences: int,
    guidance_scale: float,
    selected_species: str,
    gene_query: str,
    manual_condition_vector_csv: str,
    score_with_default_predictors: bool = True,
) -> tuple:
    enable_timing = os.getenv("GENERATOR_ENABLE_TIMING", "0") == "1"
    generation_start = time.perf_counter() if enable_timing else 0.0
    df, csv_path, fasta_path, summary = run_generation(
        checkpoint_key=checkpoint_key,
        num_sequences=num_sequences,
        guidance_scale=guidance_scale,
        selected_species=selected_species,
        gene_query=gene_query,
        manual_condition_vector_csv=manual_condition_vector_csv,
    )
    if enable_timing:
        summary["timing_generation_s"] = f"{time.perf_counter() - generation_start:.6f}"

    df["generator_species"] = summary.get("selected_species", "")
    df["generator_gene_id"] = summary.get("matched_gene_id", "")

    if score_with_default_predictors and not df.empty:
        scoring_start = time.perf_counter() if enable_timing else 0.0
        scores_by_model = score_sequences_for_models(
            sequences=df["sequence"].astype(str).tolist(),
            model_keys=DEFAULT_GENERATOR_SCORING_MODELS,
        )
        for model_key, scores in scores_by_model.items():
            df[f"prediction_score_{model_key}"] = scores
        if enable_timing:
            summary["timing_scoring_s"] = f"{time.perf_counter() - scoring_start:.6f}"

    # Ensure downloadable CSV reflects the table shown in the app.
    df.to_csv(csv_path, index=False)
    return df, csv_path, fasta_path, summary
