from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
import torch
from transformers import AutoTokenizer

from predictor_web import config
from predictor_web.models.specieslm_model import SpeciesLMLightAttention, load_checkpoint
from predictor_web.utils.fasta import parse_and_validate_fasta
from predictor_web.utils.io import write_predictions_csv


def _resolve_device() -> torch.device:
    if config.DEFAULT_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(config.DEFAULT_DEVICE)


def _sequence_to_kmers(seq: str, k: int = 6) -> list[str]:
    if len(seq) < k:
        return [seq]
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


@lru_cache(maxsize=1)
def get_tokenizer() -> Any:
    return AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)


@lru_cache(maxsize=2)
def get_loaded_model(model_key: str) -> SpeciesLMLightAttention:
    if model_key not in config.SPECIES_OPTIONS:
        raise ValueError(f"Unsupported model key: {model_key}")

    model = SpeciesLMLightAttention()
    device = _resolve_device()
    checkpoint_path = config.SPECIES_OPTIONS[model_key]["checkpoint"]
    return load_checkpoint(model, checkpoint_path, device)


def _build_model_texts(sequences: list[str], model_key: str) -> list[str]:
    species_proxy = config.SPECIES_OPTIONS[model_key]["species_proxy"]
    return [species_proxy + " " + " ".join(_sequence_to_kmers(seq, k=6)) for seq in sequences]


def _predict_texts_batched(texts: list[str], model_key: str, batch_size: int = 16) -> list[float]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    tokenizer = get_tokenizer()
    model = get_loaded_model(model_key)
    device = _resolve_device()

    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            batch_scores = model(input_ids, attention_mask).detach().cpu().tolist()
            scores.extend(float(score) for score in batch_scores)
    return scores


def predict_sequences(
    sequences: list[str],
    model_key: str,
    sequence_ids: list[str] | None = None,
    headers: list[str] | None = None,
    batch_size: int = 16,
) -> pd.DataFrame:
    if model_key not in config.SPECIES_OPTIONS:
        raise ValueError("Unsupported model. Please select 'sc' or 'pp'.")
    if not sequences:
        return pd.DataFrame(columns=["sequence_id", "header", "sequence", "selected_model", "prediction_score"])
    if sequence_ids is not None and len(sequence_ids) != len(sequences):
        raise ValueError("sequence_ids length must match sequences length.")
    if headers is not None and len(headers) != len(sequences):
        raise ValueError("headers length must match sequences length.")

    texts = _build_model_texts(sequences, model_key=model_key)
    scores = _predict_texts_batched(texts, model_key=model_key, batch_size=batch_size)

    rows = []
    for idx, (seq, score) in enumerate(zip(sequences, scores), start=1):
        rows.append(
            {
                "sequence_id": sequence_ids[idx - 1] if sequence_ids is not None else idx,
                "header": headers[idx - 1] if headers is not None else f"seq_{idx}",
                "sequence": seq,
                "selected_model": model_key,
                "prediction_score": score,
            }
        )
    return pd.DataFrame(rows)


def score_sequences_for_models(
    sequences: list[str], model_keys: list[str], batch_size: int = 16
) -> dict[str, list[float]]:
    scores_by_model: dict[str, list[float]] = {}
    for model_key in model_keys:
        scored_df = predict_sequences(sequences=sequences, model_key=model_key, batch_size=batch_size)
        scores_by_model[model_key] = scored_df["prediction_score"].astype(float).tolist()
    return scores_by_model


def run_prediction(fasta_path: str, model_key: str) -> tuple[pd.DataFrame, str]:
    if model_key not in config.SPECIES_OPTIONS:
        raise ValueError("Unsupported model. Please select 'sc' or 'pp'.")

    records = parse_and_validate_fasta(fasta_path, max_sequences=config.MAX_SEQUENCES)
    df = predict_sequences(
        sequences=[record.sequence for record in records],
        model_key=model_key,
        sequence_ids=[record.sequence_id for record in records],
        headers=[record.header for record in records],
    )
    csv_path = write_predictions_csv(df, config.OUTPUT_DIR)
    return df, csv_path
