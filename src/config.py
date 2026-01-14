# src/config.py
import os

class Config:
    # --- PENGATURAN PATH OTOMATIS (SUPAYA TIDAK ERROR) ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Specifies the location of the data folder
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    UNLABELED_DIR = os.path.join(DATA_DIR, 'unlabeled')
    LABELED_DIR = os.path.join(DATA_DIR, 'labeled')

    # --- AUDIO SETTINGS ---
    SAMPLE_RATE = 16000     
    DURATION = 3            
    N_SAMPLES = SAMPLE_RATE * DURATION 
    
    # --- TRAINING SETTINGS ---
    BATCH_SIZE = 8         
    NUM_WORKERS = 0         