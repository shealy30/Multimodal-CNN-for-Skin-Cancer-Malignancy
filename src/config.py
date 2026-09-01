from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

METADATA_CSV = RAW_DIR / "HAM10000_metadata.csv"
IMG_DIR_1 = RAW_DIR / "HAM10000_images_part_1"
IMG_DIR_2 = RAW_DIR / "HAM10000_images_part_2"

RANDOM_STATE = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 100
LEARNING_RATE = 1e-4