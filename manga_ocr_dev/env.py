from pathlib import Path
import os

ASSETS_PATH = Path(__file__).parent.parent / "assets"
MAIN_ROOT = r"/work/21013187/phuoc/Japanese_OCR/data"

DATA_SYNTHETIC_ROOT = Path(os.path.join(MAIN_ROOT,"synthetic")).expanduser()
#Train dir: DATA_SYNTHETIC_ROOT/train+DATA_SYNTHETIC_ROOT_VER
# for example 
# DATA_SYNTHETIC_ROOT = /work/21013187/phuoc/Japanese_OCR/data/synthetic
# DATA_SYNTHETIC_ROOT_VER = '2'
# Train dir: /work/21013187/phuoc/Japanese_OCR/data/synthetic/train2
DATA_SYNTHETIC_ROOT_VER = '22'

#SaveFolder
TRAIN_ROOT = Path(os.path.join(MAIN_ROOT,"outputs")).expanduser()

#TRAINNER CONFIG
RUN_NAME="debug",
NUM_EPOCHS=500,
LOGGING_STEPS = 100,
FP16=True,
SAVE_STEPS=200
EVAL_STEPS=200

#DATA SET CONFIG
DATALOADER_NUM_WORKERS = 2
MAX_SEQUENCE_LENGTH = 16,
BATCH_SIZE=200

#LOSS CONFIG
#SEE THE EFFECT ON manga_ocr_dev/training/losses.py
LABEL_SMOOTHING_FACTOR = 0.2

