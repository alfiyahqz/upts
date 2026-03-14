import os
from pathlib import Path

# ==================== PATHS ====================
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
PROCESSED_DIR = BASE_DIR / "processed_data"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
PROCESSED_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ==================== DATA PARAMETERS ====================
# ⚠️ UPDATE THIS BASED ON YOUR ACTUAL CLASSES
CLASS_MAPPING = {
    'Adhd': 0,
    'Anxiety': 1,
    'Bipolar': 2,
    'Depression': 3,
    'Pstd': 4,
    'Asd': 5,
    'Ocd': 6,
    'Eating': 7,
    'Control': 8,
    'Schizophrenia': 9
}

# ⚠️ OR if your classes are lowercase:
# CLASS_MAPPING = {
#     'adhd': 0,
#     'anxiety': 1,
#     'bipolar': 2,
#     'depression': 3,
# }

NUM_CLASSES = len(CLASS_MAPPING)

# Time series parameters
MAX_SEQUENCE_LENGTH = 100  # Maximum number of posts per user
TEXT_EMBEDDING_DIM = 768   # Sentence transformer output dimension
TEMPORAL_FEATURES_DIM = 3  # time_gap, hour_of_day, day_of_week
FEATURE_DIM = TEXT_EMBEDDING_DIM + TEMPORAL_FEATURES_DIM  # 771

# ==================== MODEL PARAMETERS ====================
EMBEDDING_MODEL_NAME = "paraphrase-xlm-r-multilingual-v1"

# Train/Val/Test split
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
RANDOM_SEED = 2022

# ==================== TRAINING PARAMETERS ====================
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3

# Model architectures to test
MODEL_LIST = [
    "fcn",
    "mcnn", 
    "inception",
    "lstm"
]

# ==================== CSV FORMAT ====================
# Column names in your CSV files
CSV_COLUMNS = {
    'class': 'class',
    'tweet_id': 'tweet_id',
    'day': 'day',
    'time': 'time',
    'text': 'tweet',  # ← The tweet text column
    'user_id': 'user_id',
    'favorite_count': 'tweet_favorite_count',
    'retweet_count': 'tweet_retweet_count',
    'source': 'tweet_source',
    'followers': 'user_followers_count',
    'friends': 'user_friends_count',
    'listed': 'user_listed_count',
    'statuses': 'user_statuses_count'
}

# Timestamp will be created by combining 'day' and 'time'
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# ==================== TEXT CLEANING ====================
# Whether to remove emojis (set False to keep them)
REMOVE_EMOJIS = False  # Keep emojis as they might be emotionally relevant

# Whether to remove mentions like @USER
REMOVE_MENTIONS = True

# Whether to remove URLs
REMOVE_URLS = True

print(f"✅ Config loaded")
print(f"   - Dataset dir: {DATASET_DIR}")
print(f"   - Number of classes: {NUM_CLASSES}")
print(f"   - Classes: {list(CLASS_MAPPING.keys())}")
print(f"   - Feature dimension: {FEATURE_DIM}")
print(f"   - Max sequence length: {MAX_SEQUENCE_LENGTH}")