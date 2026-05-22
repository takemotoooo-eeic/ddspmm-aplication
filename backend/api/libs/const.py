import os

URMP_LOUDNESS_SCORE_SILENCE = -60.0
URMP_LOUDNESS_SCORE_LOUD = -40.0

DEFAULT_SAMPLING_RATE = 16000
DEFAULT_BLOCK_SIZE = 512

# DDSP
MODEL_WEIGHTS_PATH = "api/models/ddsp/weights/last_model.pth"
PRETRAIN_CONFIG_PATH = "api/config/pretrain.config.yaml"
PREPROCESS_CONFIG_PATH = "api/config/preprocess.config.yaml"
TRAIN_CONFIG_PATH = "api/config/train.config.yaml"
LOUDNESS_PATH = "api/models/ddsp/statistics/loudness.json"

# Diffusion
DIFFUSION_MODEL_DIR = "api/models/diffusion"
DIFFUSION_MODEL_PATH = os.path.join(DIFFUSION_MODEL_DIR, "models", "last_model.pth")
DIFFUSION_CONFIG_PATH = os.path.join(DIFFUSION_MODEL_DIR, "hydra_config.yaml")
DIFFUSION_STATISTICS_PATH = os.path.join(DIFFUSION_MODEL_DIR, "statistics", "diffusion_statistics.json")
INSTRUMENT_MAPPING_PATH = os.path.join(DIFFUSION_MODEL_DIR, "instrument_mapping.json")

# FluidSynth
FLUIDSYNTH_ASSETS_DIR = "api/models/fluidsynth_assets"
FLUIDSYNTH_SOUNDFONT_PATH = os.path.join(
    FLUIDSYNTH_ASSETS_DIR,
    "soundfonts",
    os.environ.get("SOUNDFONT_FILENAME", "FluidR3_GM.sf2"),
)
FLUIDSYNTH_GM_PROGRAM_MAPPING_PATH = os.path.join(
    FLUIDSYNTH_ASSETS_DIR, "gm_program_mapping.json"
)
