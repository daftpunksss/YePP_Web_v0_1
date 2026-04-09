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
  - Select species (loaded from cfg species-reference table)
  - Enter gene name / gene_id for cfg-based condition lookup
  - Optionally provide a manual condition vector override (advanced)
  - Run Dirichlet flow matching generation
  - Automatically score generated sequences with default predictor models (`sc`, `pp`) in the same workflow (toggleable)
  - View condition summary (species, requested gene, matched gene_id, resolved yes/no, condition dimension)
  - View generated sequences preview and predictor scores
  - Download CSV (including scores) and FASTA

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
- `YEPP_CFG_SPECIES_TABLE` (default: `./data/lianlab_aval_yeast_promoter_gene_info.csv`)
- `YEPP_CFG_GENE_CONDITION_TABLE` (default: `./data/codon_ga.csv`)

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
  - Species dropdown values are loaded dynamically from `YEPP_CFG_SPECIES_TABLE` and sorted exactly as cfg usage.
  - Gene lookup is validated against `YEPP_CFG_GENE_CONDITION_TABLE` using selected species + gene identifier.
  - Missing matches and ambiguous matches return user-facing errors.
  - Missing required codon columns (64 fixed codons) return a clear error.
  - Resolved condition is assembled as `[codon_64, species_one_hot_sorted]` and validated against `YEPP_GENERATOR_CONDITION_DIM`.
  - Manual condition override (if provided) must be exactly `YEPP_GENERATOR_CONDITION_DIM` comma-separated numeric values.
