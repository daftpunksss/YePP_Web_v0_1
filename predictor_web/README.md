# Predictor Web (YePP Phase 1)

Minimal Gradio app for predictor inference on uploaded FASTA files.

## Scope
- Upload FASTA
- Select one model: `sc` (S. cerevisiae) or `pp` (P. pastoris)
- Run predictor inference
- View results table
- Download CSV

## Environment variables
- `YEPP_SC_CHECKPOINT` (default: `./checkpoints/sc_ft_last2layers.ckpt`)
- `YEPP_PP_CHECKPOINT` (default: `./checkpoints/pp_ft_last2layers.ckpt`)
- `YEPP_PREDICTOR_DEVICE` (default: `cuda`; automatically falls back to CPU when CUDA is unavailable)
- `YEPP_OUTPUT_DIR` (default: `./predictor_outputs`)
- `YEPP_MAX_SEQUENCES` (default: `200`)
- `YEPP_SPECIESLM_MODEL_ID` (default: `gagneurlab/SpeciesLM`)
- `YEPP_SPECIESLM_REVISION` (default: `upstream_species_lm`)

## Run locally
```bash
cd /workspace/YePP_Web_v0_1
python -m venv .venv
source .venv/bin/activate
pip install -r predictor_web/requirements.txt
export YEPP_SC_CHECKPOINT=/path/to/sc_ft_last2layers.ckpt
export YEPP_PP_CHECKPOINT=/path/to/pp_ft_last2layers.ckpt
python -m predictor_web.app
```

Then open the local Gradio URL shown in terminal.

## Input validation
- File must not be empty.
- FASTA structure is required (`>` header lines).
- Empty sequences are rejected.
- Only DNA letters `A/C/G/T/N` are accepted (case-insensitive).
- Upload is rejected if sequence count exceeds `YEPP_MAX_SEQUENCES`.
