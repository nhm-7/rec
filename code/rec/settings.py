from pathlib import Path

PROJECT_PATH = str(Path(__file__).resolve().parents[2])
TRANSFORMER_MODEL = 'bert-base-uncased'
EPS = 1e-5
MODELS_OUTPUT = f"{PROJECT_PATH}/models"