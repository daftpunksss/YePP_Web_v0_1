# Predictor Web (YePP Phase 1 + Generator)

Minimal Gradio app for predictor inference and generator inference.

## Scope
- Predictor tab
  - Upload FASTA
  - Select one model: `sc` (S. cerevisiae) or `pp` (P. pastoris)
  - Run predictor inference
  - View results table
  - Download CSV
- Generator tab
  - Select generator checkpoint
  - Set number of sequences and guidance scale
  - Optionally provide a condition vector
  - Run Dirichlet flow matching generation
  - View generated sequences preview
  - Download CSV and FASTA

## Environment variables
### Predictor
- `YEPP_SC_CHECKPOINT` (default: `./checkpoints/sc_ft_last2layers.ckpt`)
- `YEPP_PP_CHECKPOINT` (default: `./checkpoints/pp_ft_last2layers.ckpt`)
- `YEPP_PREDICTOR_DEVICE` (default: `cuda`; automatically falls back to CPU when CUDA is unavailable)
- `YEPP_OUTPUT_DIR` (default: `./predictor_outputs`)
- `YEPP_MAX_SEQUENCES` (default: `200`)
- `YEPP_SPECIESLM_MODEL_ID` (default: `gagneurlab/SpeciesLM`)
- `YEPP_SPECIESLM_REVISION` (default: `upstream_species_lm`)

### Generator
- `YEPP_GENERATOR_CHECKPOINT` (default: `./checkpoints/generator_dirichlet.ckpt`)
- `YEPP_GENERATOR_CHECKPOINTS` (optional CSV list of checkpoint paths; enables dropdown model selection)
- `YEPP_GENERATOR_DEVICE` (default: same as predictor device)
- `YEPP_GENERATOR_OUTPUT_DIR` (default: `./generator_outputs`)
- `YEPP_GENERATOR_SEQUENCE_LENGTH` (default: `500`)
- `YEPP_GENERATOR_CONDITION_DIM` (default: `84`)
- `YEPP_GENERATOR_MAX_SEQUENCES` (default: `128`)
- `YEPP_GENERATOR_PRIOR_PSEUDOCOUNT` (default: `2.0`)
- `YEPP_GENERATOR_ALPHA_MAX` (default: `8.0`)
- `YEPP_GENERATOR_NUM_STEPS` (default: `400`)
- `YEPP_GENERATOR_FLOW_TEMP` (default: `1.0`)
- `YEPP_GENERATOR_GUIDANCE_SCALE` (default: `1.0`)
- `YEPP_GENERATOR_USE_MIXED_PRECISION` (`1` to enable CUDA autocast, default: `0`)

## Run locally
```bash
cd /workspace/YePP_Web_v0_1
python -m venv .venv
source .venv/bin/activate
pip install -r predictor_web/requirements.txt
export YEPP_SC_CHECKPOINT=/path/to/sc_ft_last2layers.ckpt
export YEPP_PP_CHECKPOINT=/path/to/pp_ft_last2layers.ckpt
export YEPP_GENERATOR_CHECKPOINT=/path/to/generator_dirichlet.ckpt
python -m predictor_web.app
```

Then open the local Gradio URL shown in terminal.

## Input validation
- Predictor:
  - File must not be empty.
  - FASTA structure is required (`>` header lines).
  - Empty sequences are rejected.
  - Only DNA letters `A/C/G/T/N` are accepted (case-insensitive).
  - Upload is rejected if sequence count exceeds `YEPP_MAX_SEQUENCES`.
- Generator:
  - `num_sequences` must be between `1` and `YEPP_GENERATOR_MAX_SEQUENCES`.
  - Condition vector must be exactly `YEPP_GENERATOR_CONDITION_DIM` comma-separated numeric values (or empty).
