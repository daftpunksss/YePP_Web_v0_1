from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
import torch
from transformers import AutoTokenizer

from predictor_web import config
from predictor_web.models.specieslm_model import SpeciesLMLightAttention, load_checkpoint
from predictor_web.utils.fasta import FastaRecord, parse_and_validate_fasta
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


def _predict_single(record: FastaRecord, model_key: str) -> float:
    species_proxy = config.SPECIES_OPTIONS[model_key]["species_proxy"]
    text = species_proxy + " " + " ".join(_sequence_to_kmers(record.sequence, k=6))

    tokenizer = get_tokenizer()
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    device = _resolve_device()

    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    model = get_loaded_model(model_key)
    with torch.no_grad():
        score = model(input_ids, attention_mask).item()
    return float(score)


def run_prediction(fasta_path: str, model_key: str) -> tuple[pd.DataFrame, str]:
    if model_key not in config.SPECIES_OPTIONS:
        raise ValueError("Unsupported model. Please select 'sc' or 'pp'.")

    records = parse_and_validate_fasta(fasta_path, max_sequences=config.MAX_SEQUENCES)

    rows = []
    for record in records:
        score = _predict_single(record, model_key)
        rows.append(
            {
                "sequence_id": record.sequence_id,
                "header": record.header,
                "sequence": record.sequence,
                "selected_model": model_key,
                "prediction_score": score,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = write_predictions_csv(df, config.OUTPUT_DIR)
    return df, csv_path
