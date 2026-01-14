# src/config.py
import os

class Config:
    # --- Audio Settings ---
    # The sample rate of 22050 Hz is a common standard for speech/audio classification.
    SAMPLE_RATE = 22050
    
    # Audio cut duration (in seconds). 
    DURATION = 3 
    
    # Number of samples per file = Sample Rate * Duration
    N_SAMPLES = SAMPLE_RATE * DURATION
    
    # --- Spectrogram Settings ---
    # N_MELS: Spectrogram image height 
    N_MELS = 128
    
    # N_FFT dan HOP_LENGTH for Short-Time Fourier Transform (STFT)
    N_FFT = 2048
    HOP_LENGTH = 512
    
    # --- Training Settings ---
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 10
    NUM_CLASSES = 5