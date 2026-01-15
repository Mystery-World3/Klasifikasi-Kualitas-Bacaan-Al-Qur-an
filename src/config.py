# src/config.py

class Config:
    # --- Audio Settings ---
    SAMPLE_RATE = 22050
    DURATION = 30            
    N_SAMPLES = SAMPLE_RATE * DURATION
    
    # --- Spectrogram Settings ---
    N_MELS = 128            
    N_FFT = 2048
    HOP_LENGTH = 512
    
    # --- Training Settings ---
    BATCH_SIZE = 16         # Kurangi jika memory error
    LEARNING_RATE = 0.001
    EPOCHS_PRETRAIN = 5     # Epoch untuk Stage 1 (Sedikit saja untuk demo)
    EPOCHS_FINETUNE = 20    # Epoch untuk Stage 2
    
    # --- Path Data ---
    UNLABELED_DIR = "data/unlabeled"
    LABELED_DIR = "data/labeled"