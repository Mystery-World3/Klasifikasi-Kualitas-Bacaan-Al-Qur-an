# src/utils.py
import librosa
import numpy as np
import torch
from .config import Config

class AudioUtil:
    @staticmethod
    def open_audio(audio_path):
        """
        Membuka file audio dan melakukan resampling ke target SAMPLE_RATE.
        Referensi: https://librosa.org/doc/main/generated/librosa.load.html
        """
        try:
            # Load audio, automatically resample, and convert to mono
            sig, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True)
            return sig
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None

    @staticmethod
    def pad_trunc(sig, max_len):
        """
        Memotong audio jika terlalu panjang, atau menambah 'padding' (hening) jika terlalu pendek.
        Tujuannya agar dimensi input ke AI selalu konsisten.
        """
        sig_len = len(sig)
        
        if sig_len > max_len:
            # Truncate (Potong kelebihan)
            sig = sig[:max_len]
        elif sig_len < max_len:
            # Pad (Tambah nol di belakang)
            pad_len = max_len - sig_len
            sig = np.pad(sig, (0, pad_len), mode='constant')
            
        return sig

    @staticmethod
    def audio_to_melspectrogram(sig):
        """
        Mengubah gelombang suara (Time Domain) menjadi Mel Spectrogram (Frequency Domain).
        Ini adalah fitur standar dalam Deep Learning for Audio.
        Referensi: https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
        """
        # 1. Hitung Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=sig, 
            sr=Config.SAMPLE_RATE, 
            n_mels=Config.N_MELS,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH
        )
        
        # 2. Convert ke Log Scale (dB) karena telinga manusia mendengar secara logaritmik
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 3. Normalisasi Min-Max ke range 0-1 (Agar AI lebih cepat belajar)
        norm_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-6)
        
        return norm_spec

    @staticmethod
    def preprocess(audio_path):
        """
        Pipeline lengkap: Load -> Pad -> Spectrogram -> Tensor
        Output: Tensor dengan dimensi (1, n_mels, time_steps)
        """
        sig = AudioUtil.open_audio(audio_path)
        if sig is None:
            return None
            
        sig = AudioUtil.pad_trunc(sig, Config.N_SAMPLES)
        spec = AudioUtil.audio_to_melspectrogram(sig)
        
        # Convert numpy array to PyTorch Tensor
        # Add channel dimension (1) in front -> (Channel, Freq, Time)
        tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        return tensor