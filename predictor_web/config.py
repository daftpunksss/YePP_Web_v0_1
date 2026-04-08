import os
from pathlib import Path

MODEL_ID = os.getenv("YEPP_SPECIESLM_MODEL_ID", "gagneurlab/SpeciesLM")
MODEL_REVISION = os.getenv("YEPP_SPECIESLM_REVISION", "upstream_species_lm")

SC_CHECKPOINT_PATH = os.getenv("YEPP_SC_CHECKPOINT", "./checkpoints/sc_ft_last2layers.ckpt")
PP_CHECKPOINT_PATH = os.getenv("YEPP_PP_CHECKPOINT", "./checkpoints/pp_ft_last2layers.ckpt")

DEFAULT_DEVICE = os.getenv("YEPP_PREDICTOR_DEVICE", "cuda")
MAX_SEQUENCES = int(os.getenv("YEPP_MAX_SEQUENCES", "200"))

OUTPUT_DIR = Path(os.getenv("YEPP_OUTPUT_DIR", "./predictor_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES_OPTIONS = {
    "sc": {
        "label": "S. cerevisiae",
        "species_proxy": "kazachstania_africana_cbs_2517_gca_000304475",
        "checkpoint": SC_CHECKPOINT_PATH,
    },
    "pp": {
        "label": "P. pastoris",
        "species_proxy": "komagataella_phaffii_gs115_gca_001746955",
        "checkpoint": PP_CHECKPOINT_PATH,
    },
}

ALLOWED_DNA_CHARS = set("ACGTN")


GENERATOR_DEVICE = os.getenv("YEPP_GENERATOR_DEVICE", DEFAULT_DEVICE)
GENERATOR_MODE = os.getenv("YEPP_GENERATOR_MODE", "dirichlet")
GENERATOR_SEQUENCE_LENGTH = int(os.getenv("YEPP_GENERATOR_SEQUENCE_LENGTH", "500"))
GENERATOR_CONDITION_DIM = int(os.getenv("YEPP_GENERATOR_CONDITION_DIM", "84"))
GENERATOR_MAX_SEQUENCES = int(os.getenv("YEPP_GENERATOR_MAX_SEQUENCES", "128"))
GENERATOR_PRIOR_PSEUDOCOUNT = float(os.getenv("YEPP_GENERATOR_PRIOR_PSEUDOCOUNT", "2.0"))
GENERATOR_ALPHA_MAX = float(os.getenv("YEPP_GENERATOR_ALPHA_MAX", "8.0"))
GENERATOR_NUM_INTEGRATION_STEPS = int(os.getenv("YEPP_GENERATOR_NUM_STEPS", "400"))
GENERATOR_FLOW_TEMP = float(os.getenv("YEPP_GENERATOR_FLOW_TEMP", "1.0"))
GENERATOR_GUIDANCE_SCALE = float(os.getenv("YEPP_GENERATOR_GUIDANCE_SCALE", "1.0"))
GENERATOR_USE_MIXED_PRECISION = os.getenv("YEPP_GENERATOR_USE_MIXED_PRECISION", "0") == "1"

GENERATOR_OUTPUT_DIR = Path(os.getenv("YEPP_GENERATOR_OUTPUT_DIR", "./generator_outputs"))
GENERATOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_generator_checkpoints() -> dict[str, Path]:
    csv_value = os.getenv("YEPP_GENERATOR_CHECKPOINTS", "").strip()
    if csv_value:
        paths = [Path(item.strip()) for item in csv_value.split(",") if item.strip()]
        return {path.stem: path for path in paths}

    single = Path(os.getenv("YEPP_GENERATOR_CHECKPOINT", "./checkpoints/generator_dirichlet.ckpt"))
    return {single.stem: single}


GENERATOR_CHECKPOINTS = _resolve_generator_checkpoints()
