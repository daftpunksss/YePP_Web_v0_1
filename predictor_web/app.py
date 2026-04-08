from __future__ import annotations

import gradio as gr
import pandas as pd

from predictor_web.config import (
    GENERATOR_CHECKPOINTS,
    GENERATOR_CONDITION_DIM,
    GENERATOR_GUIDANCE_SCALE,
    GENERATOR_MAX_SEQUENCES,
    MAX_SEQUENCES,
    SPECIES_OPTIONS,
)
from predictor_web.services.generator_service import run_generation
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


def _run_generator(checkpoint_key: str, num_sequences: int, guidance_scale: float, cond_vector: str):
    try:
        df, csv_path, fasta_path = run_generation(
            checkpoint_key=checkpoint_key,
            num_sequences=int(num_sequences),
            guidance_scale=float(guidance_scale),
            condition_vector_csv=cond_vector,
        )
        msg = (
            f"Generation completed with checkpoint '{checkpoint_key}'. "
            f"Generated {len(df)} sequence(s)."
        )
        return df, csv_path, fasta_path, msg
    except FileNotFoundError as err:
        return pd.DataFrame(), None, None, f"Generator checkpoint error: {err}"
    except ValueError as err:
        return pd.DataFrame(), None, None, f"Input validation error: {err}"
    except Exception as err:
        return pd.DataFrame(), None, None, f"Unexpected error during generation: {err}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="YePP Predictor + Generator") as demo:
        gr.Markdown("# YePP Local App")

        with gr.Tabs():
            with gr.Tab("Predictor (Phase 1)"):
                gr.Markdown(
                    """
                    Upload a FASTA file, select one predictor model, and run local inference.
                    Supported models: **S. cerevisiae (sc)** and **P. pastoris (pp)**.
                    """
                )

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

                gr.Markdown(
                    f"Validation note: up to {MAX_SEQUENCES} sequences per upload; allowed DNA letters are A/C/G/T/N."
                )

                run_button.click(
                    fn=_run_predict,
                    inputs=[fasta_input, model_input],
                    outputs=[table_output, csv_output, status_output],
                )

            with gr.Tab("Generator"):
                gr.Markdown(
                    """
                    Generate promoter DNA sequences using the existing Dirichlet flow matching generator.
                    Provide an optional condition vector (comma-separated floats).
                    """
                )
                checkpoint_choices = list(GENERATOR_CHECKPOINTS.keys())
                checkpoint_input = gr.Dropdown(
                    choices=checkpoint_choices,
                    value=checkpoint_choices[0],
                    label="Generator checkpoint",
                )
                num_sequences_input = gr.Slider(
                    minimum=1,
                    maximum=GENERATOR_MAX_SEQUENCES,
                    value=min(8, GENERATOR_MAX_SEQUENCES),
                    step=1,
                    label="Number of sequences",
                )
                guidance_input = gr.Number(value=GENERATOR_GUIDANCE_SCALE, label="Guidance scale")
                condition_input = gr.Textbox(
                    label=f"Condition vector ({GENERATOR_CONDITION_DIM} comma-separated floats; optional)",
                    placeholder="Leave empty to use a zero vector (unconditioned baseline)",
                    lines=3,
                )
                run_gen_button = gr.Button("Run generation")

                gen_status = gr.Textbox(label="Status", interactive=False)
                gen_table = gr.Dataframe(label="Generated sequences preview", interactive=False)
                gen_csv = gr.File(label="Download CSV")
                gen_fasta = gr.File(label="Download FASTA")

                run_gen_button.click(
                    fn=_run_generator,
                    inputs=[checkpoint_input, num_sequences_input, guidance_input, condition_input],
                    outputs=[gen_table, gen_csv, gen_fasta, gen_status],
                )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
