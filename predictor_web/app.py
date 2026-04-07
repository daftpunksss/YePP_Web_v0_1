from __future__ import annotations

import gradio as gr
import pandas as pd

from predictor_web.config import MAX_SEQUENCES, SPECIES_OPTIONS
from predictor_web.services.predict_service import run_prediction
from predictor_web.utils.fasta import FastaValidationError


def _run_predict(file_obj, model_key: str):
    if file_obj is None:
        return pd.DataFrame(), None, "Please upload a FASTA file before running prediction."

    try:
        df, csv_path = run_prediction(file_obj.name, model_key)
        model_label = SPECIES_OPTIONS[model_key]["label"]
        message = f"Prediction completed with {model_label}. Processed {len(df)} sequence(s)."
        return df, csv_path, message
    except FastaValidationError as err:
        return pd.DataFrame(), None, f"Input validation error: {err}"
    except FileNotFoundError as err:
        return pd.DataFrame(), None, f"Model checkpoint error: {err}"
    except Exception as err:  # fallback for local development visibility
        return pd.DataFrame(), None, f"Unexpected error during inference: {err}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="YePP Predictor (Phase 1)") as demo:
        gr.Markdown("""
        # YePP Predictor (Phase 1)
        Upload a FASTA file, select one predictor model, and run local inference.
        Supported models: **S. cerevisiae (sc)** and **P. pastoris (pp)**.
        """)

        fasta_input = gr.File(label="Upload FASTA file", file_types=[".fasta", ".fa", ".fna", ".txt"])
        model_input = gr.Radio(
            choices=[("S. cerevisiae (sc)", "sc"), ("P. pastoris (pp)", "pp")],
            value="sc",
            label="Select predictor model",
        )
        run_button = gr.Button("Run prediction")

        status_output = gr.Textbox(label="Status", interactive=False)
        table_output = gr.Dataframe(label="Results table", interactive=False)
        csv_output = gr.File(label="Download CSV")

        gr.Markdown(f"Validation note: up to {MAX_SEQUENCES} sequences per upload; allowed DNA letters are A/C/G/T/N.")

        run_button.click(
            fn=_run_predict,
            inputs=[fasta_input, model_input],
            outputs=[table_output, csv_output, status_output],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
