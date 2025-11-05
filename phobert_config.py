"""
Configuration file for PhoBERT NER training
"""

# ============ DATA SETTINGS ============
# Path to your ViMedNER dataset
VIMEDNER_DIR = r"./vimedner_data"  # Update this to your ViMedNER folder
TRAIN_FILE = "train.txt"
DEV_FILE = "dev.txt"
TEST_FILE = "test.txt"

# If you only have one file, set SINGLE_FILE = True
SINGLE_FILE = True  # Set to True if you have only one .txt file
SINGLE_FILE_PATH = r"./vimedner_data/train.txt"  # Update if SINGLE_FILE=True

# Train/Dev/Test split ratios (only used if SINGLE_FILE=True)
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

# ============ MODEL SETTINGS ============
PHOBERT_MODEL = "vinai/phobert-base"  # or "vinai/phobert-large" for better quality
MAX_LENGTH = 256  # Maximum sequence length (adjust based on your text chunks)

# Entity types to extract (maps from ViMedNER to your categories)
# ViMedNER tags -> Your categories (actual tags from the dataset)
ENTITY_MAPPING = {
    "ten_benh": "Bệnh",
    "trieu_chung_benh": "Triệu chứng",
    "nguyen_nhan_benh": "Nguyên nhân",
    "bien_phap_chan_doan": "Chẩn đoán",
    "bien_phap_dieu_tri": "Điều trị",
    # Note: No "Thuốc", "Phòng ngừa", "Biến chứng" in ViMedNER data
}

# BIO tags that will be created
# Example: B-ten_benh, I-ten_benh, B-trieu_chung, I-trieu_chung, B-thuoc, I-thuoc, O
LABEL_LIST = None  # Will be auto-generated from data

# ============ TRAINING SETTINGS ============
OUTPUT_DIR = r"./phobert_medical_ner"  # Where to save trained model
BATCH_SIZE = 16  # Reduce to 8 if GPU memory issues, increase to 32 if you have big GPU
LEARNING_RATE = 5e-5  # Standard for BERT fine-tuning
NUM_EPOCHS = 3  # Usually 3-5 epochs is enough
WARMUP_RATIO = 0.1  # Gradual learning rate warmup
WEIGHT_DECAY = 0.01

# Evaluation settings
EVAL_STEPS = 500  # Evaluate every N steps
SAVE_STEPS = 500  # Save checkpoint every N steps
LOGGING_STEPS = 100  # Log metrics every N steps

# Early stopping
EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for N evaluations

# ============ HARDWARE SETTINGS ============
USE_GPU = True  # Set to False if you want CPU only
FP16 = True  # Use mixed precision training (faster, less memory) - requires GPU

# ============ INFERENCE SETTINGS ============
INFERENCE_BATCH_SIZE = 32  # Batch size for prediction
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to accept entity (0-1)

# ============ OUTPUT FORMAT ============
# Match your existing NER_extractor.py format
NER_KEYS = [
    "Bệnh",
    "Triệu chứng",
    "Nguyên nhân",
    "Chẩn đoán",
    "Điều trị",
]
