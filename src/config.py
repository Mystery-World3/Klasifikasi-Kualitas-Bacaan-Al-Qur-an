import os
import random
import numpy as np
import torch


class Config:

    # RANDOM SEED
    DEFAULT_SEED = 42
    SEEDS = [
        42,
        52,
        62,
        72,
        82
    ]

    # DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # AUDIO
    SAMPLE_RATE = 22050
    DURATION = 3
    N_SAMPLES = SAMPLE_RATE * DURATION

    # MEL SPECTROGRAM
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    F_MIN = 20
    F_MAX = 8000

    # CONTRASTIVE
    EMBEDDING_DIM = 128
    TEMPERATURE = 0.5

    # TRAINING
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS_PRETRAIN = 30
    EPOCHS_FINETUNE = 30
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # EARLY STOPPING
    EARLY_STOPPING_PATIENCE = 5

    # LR SCHEDULER
    LR_FACTOR = 0.1
    LR_PATIENCE = 3

    # DATASET
    UNLABELED_DIR = "data/unlabeled"
    LABELED_DIR = "data/labeled"

    # OUTPUT
    MODEL_DIR = "models"
    RESULT_DIR = "results"
    HISTORY_DIR = "history"
    FIGURE_DIR = "figures"

    # SAVE FILE
    PRETRAIN_MODEL = "encoder_pretrained"
    FINAL_MODEL = "final_model"

    # SAVE OPTION
    SAVE_HISTORY = True
    SAVE_MODEL = True
    SAVE_CONFUSION_MATRIX = True
    SAVE_TSNE = True

    # CREATE FOLDER
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)


# SET RANDOM SEED
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False