# src/utils.py
import librosa
import numpy as np
import torch
from .config import Config

class AudioUtil:
    @staticmethod
    def open_audio(audio_path):
        """Load audio dan resample."""
        try:
            sig, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True)
            return sig
        except Exception as e:
            # print(f"Error loading {audio_path}: {e}")
            return None

    @staticmethod
    def pad_trunc(sig, max_len):
        """Potong atau tambah padding agar panjang array konsisten."""
        if sig is None: return np.zeros(max_len)
        
        sig_len = len(sig)
        if sig_len > max_len:
            sig = sig[:max_len]
        elif sig_len < max_len:
            pad_len = max_len - sig_len
            sig = np.pad(sig, (0, pad_len), mode='constant')
        return sig

    @staticmethod
    def audio_to_melspectrogram(sig):
        """Ubah audio menjadi gambar Mel-Spectrogram."""
        spec = librosa.feature.melspectrogram(
            y=sig, 
            sr=Config.SAMPLE_RATE, 
            n_mels=Config.N_MELS,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH
        )
        # Convert to Log Scale (dB) dan Normalisasi 0-1
        log_spec = librosa.power_to_db(spec, ref=np.max)
        norm_spec = (log_spec - log_spec.min()) / (log_spec.max() - log_spec.min() + 1e-6)
        return norm_spec

    @staticmethod
    def preprocess(audio_path, add_noise=False):
        """Pipeline lengkap: File -> Tensor."""
        sig = AudioUtil.open_audio(audio_path)
        if sig is None: return None
        
        sig = AudioUtil.pad_trunc(sig, Config.N_SAMPLES)
        
        # Simple Augmentation for Contrastive Learning (Stage 1)
        if add_noise:
            noise = np.random.randn(len(sig)) * 0.005
            sig = sig + noise

        spec = AudioUtil.audio_to_melspectrogram(sig)
        
        # Output Shape: (1, n_mels, time)
        tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        return tensor