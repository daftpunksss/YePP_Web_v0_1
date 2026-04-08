from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_generation_dataframe(sequences: list[str], checkpoint_name: str) -> pd.DataFrame:
    rows = []
    for idx, seq in enumerate(sequences, start=1):
        rows.append(
            {
                "sequence_id": f"generated_{idx:04d}",
                "generator_checkpoint": checkpoint_name,
                "sequence": seq,
                "length": len(seq),
                "gc_content": (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_generation_csv(df: pd.DataFrame, output_dir: Path, stem: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.csv"
    df.to_csv(path, index=False)
    return str(path)


def write_generation_fasta(df: pd.DataFrame, output_dir: Path, stem: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.fasta"
    with path.open("w", encoding="utf-8") as handle:
        for row in df.itertuples(index=False):
            handle.write(f">{row.sequence_id}|{row.generator_checkpoint}\n")
            handle.write(f"{row.sequence}\n")
    return str(path)
