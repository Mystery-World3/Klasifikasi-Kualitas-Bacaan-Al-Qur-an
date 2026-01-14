# src/config.py
import os

class Config:
    # --- PENGATURAN PATH OTOMATIS (SUPAYA TIDAK ERROR) ---
    # Mendapatkan lokasi file config.py ini, lalu mundur satu folder ke belakang (folder project)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Menentukan lokasi folder data secara absolut
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    UNLABELED_DIR = os.path.join(DATA_DIR, 'unlabeled')
    LABELED_DIR = os.path.join(DATA_DIR, 'labeled')

    # --- AUDIO SETTINGS ---
    SAMPLE_RATE = 16000     
    DURATION = 3            # Kita coba pendekkan jadi 3 detik dulu
    N_SAMPLES = SAMPLE_RATE * DURATION 
    
    # --- TRAINING SETTINGS ---
    BATCH_SIZE = 8          # Kecilkan dulu untuk tes di laptop
    NUM_WORKERS = 0         # Set 0 dulu agar stabil di Windows