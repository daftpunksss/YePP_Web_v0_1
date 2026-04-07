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
