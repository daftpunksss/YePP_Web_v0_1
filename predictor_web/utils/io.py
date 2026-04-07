from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def write_predictions_csv(df: pd.DataFrame, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"predictor_results_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    return str(output_path)
