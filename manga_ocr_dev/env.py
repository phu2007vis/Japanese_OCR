from pathlib import Path
import os
ASSETS_PATH = Path(__file__).parent.parent / "assets"

MAIN_ROOT = r"/work/21013187/phuoc/Japanese_OCR/data"
# FONTS_ROOT = Path(os.path.join(MAIN_ROOT,"fonts")).expanduser()
DATA_SYNTHETIC_ROOT = Path(os.path.join(MAIN_ROOT,"synthetic")).expanduser()
DATA_SYNTHETIC_ROOT_VER = '2'

BACKGROUND_DIR = Path(os.path.join(MAIN_ROOT,"backgrounds")).expanduser()
# MANGA109_ROOT = Path("~/data/manga/Manga109s").expanduser()
TRAIN_ROOT = Path(os.path.join(MAIN_ROOT,"outputs")).expanduser()
ORIGINAL_BACKGROUND = Path(os.path.join(MAIN_ROOT,"original_background")).expanduser()


run_name="debug",
max_length = 16,
batch_size=200,
num_epochs=500,
logging_steps = 100,
fp16=True,
save_steps=200,
eval_steps=200,
