from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

LOG_DIR = PROJECT_ROOT / "logs"

NUMBER_OF_PARTICIPANTS = 500
NUMBER_OF_DAYS = 365
RANDOM_SEED = 42

PARTICIPANT_FILE = RAW_DIR / "participant_profiles.csv"
DAILY_DATA_FILE = RAW_DIR / "daily_observations.csv"
WEATHER_FILE = RAW_DIR / "weather.csv"
FINAL_DATASET_FILE = RAW_DIR / "PaceMate_dataset_v3.csv"

DATA_DICTIONARY_FILE = METADATA_DIR / "data_dictionary.csv"

TRAIN_FILE = PROCESSED_DIR / "train.csv"
VALIDATION_FILE = PROCESSED_DIR / "validation.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

MIN_AGE = 16
MAX_AGE = 65

MIN_HEIGHT_CM = 145
MAX_HEIGHT_CM = 200

MIN_WEIGHT_KG = 40
MAX_WEIGHT_KG = 150

MIN_TEMP = -5
MAX_TEMP = 42

MIN_HUMIDITY = 15
MAX_HUMIDITY = 100

RESTING_HR_MIN = 45
RESTING_HR_MAX = 95

STANDING_HR_DELTA_MIN = 10
STANDING_HR_DELTA_MAX = 60

HRV_MIN = 15
HRV_MAX = 120

MIN_SLEEP = 3.0
MAX_SLEEP = 10.5

MIN_WATER = 750
MAX_WATER = 4500

MIN_STEPS = 0
MAX_STEPS = 25000

MAX_ACTIVE_MINUTES = 300
MAX_STANDING_MINUTES = 900

SYMPTOM_MIN = 0
SYMPTOM_MAX = 10

FLARE_LABELS = [
    "Low",
    "Medium",
    "High"
]

HYDRATION_LABELS = [
    "Low",
    "Moderate",
    "High"
]

REST_LABELS = [
    "Normal Activity",
    "Consider Pacing",
    "Prioritize Rest"
]