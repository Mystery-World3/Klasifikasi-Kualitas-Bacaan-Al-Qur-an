import os
import random
import numpy as np
import torch

if os.path.exists("/kaggle/input"):
    BASE_DATASET = "/kaggle/input/datasets/raffaarvel/dataset-maryam-1-10-potong/Dataset Tahsin 3"
else:
    BASE_DATASET = "data"

class Config:
    DEFAULT_SEED = 42
    SEEDS = [42, 52, 62, 72, 82]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    SAMPLE_RATE = 22050
    DURATION = 3
    N_SAMPLES = SAMPLE_RATE * DURATION
    
    N_MELS = 64
    N_FFT = 2048
    HOP_LENGTH = 512
    F_MIN = 20
    F_MAX = 8000

    EMBEDDING_DIM = 128
    TEMPERATURE = 0.5

    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS_PRETRAIN = 30
    EPOCHS_FINETUNE = 30

    NUM_WORKERS = 4
    PIN_MEMORY = True

    EARLY_STOPPING_PATIENCE = 5
    LR_FACTOR = 0.1
    LR_PATIENCE = 3

    LABELED_DIR = os.path.join(BASE_DATASET, "Label")
    UNLABELED_DIR = os.path.join(BASE_DATASET, "unlabeled")

    MODEL_DIR = "models"
    RESULT_DIR = "results"
    HISTORY_DIR = "history"
    FIGURE_DIR = "figures"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False