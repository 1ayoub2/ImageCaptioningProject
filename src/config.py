import os
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGE_DIR = RAW_DIR / "Images"
CAPTION_FILE = RAW_DIR / "captions.txt"
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"

for d in (CHECKPOINT_DIR, LOG_DIR, FIGURE_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
ENCODED_IMAGE_SIZE = 14
MIN_WORD_FREQ = 5
CAPTION_MAX_LEN = 40
D_MODEL = 512
NUM_HEADS = 8
NUM_DECODER_LAYERS = 6
DIM_FF = 2048
DROPOUT = 0.5
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 2000
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
FREEZE_CNN_EPOCHS = 10
BETAS = (0.9, 0.98)
EPS = 1e-9
BEAM_WIDTH = 3
MAX_GEN_LEN = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
# Disjoint splits by image id (fractions must sum to 1.0)
SPLIT_SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
# Validation BLEU/METEOR during training (beam search on val set). 1 = every epoch; 0 = off.
VAL_METRICS_EVERY_N_EPOCHS = 1
# Full training state for resume / Colab (overwritten each epoch). Not for GitHub (too large).
SAVE_RESUME_EVERY_EPOCH = True
RESUME_CHECKPOINT_FILENAME = "last.pt"
