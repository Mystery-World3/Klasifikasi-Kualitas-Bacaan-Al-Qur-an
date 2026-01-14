# src/utils.py
import librosa
import numpy as np
import torch

class AudioUtil:
    @staticmethod
    def open_audio(audio_path, target_sr):
        """Membuka file audio dan convert ke Mono"""
        sig, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return sig

    @staticmethod
    def rechannel(sig, num_channels=1):
        """Memastikan audio mono (1 channel)"""
        if len(sig.shape) == 1:
            sig = sig[np.newaxis, :]
        return sig

    @staticmethod
    def pad_trunc(sig, max_len):
        """Memotong (Truncate) atau Menambah (Pad) durasi audio agar seragam"""
        num_rows, sig_len = sig.shape
        if sig_len > max_len:
            # cut if it's too long
            sig = sig[:, :max_len]
        elif sig_len < max_len:
            # Add padding if it's too short
            pad_begin_len = np.random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = np.zeros((num_rows, pad_begin_len))
            pad_end = np.zeros((num_rows, pad_end_len))
            sig = np.concatenate((pad_begin, sig, pad_end), axis=1)
        return sig

    @staticmethod
    def time_shift(sig, shift_limit=0.4):
        """Augmentasi: Menggeser waktu audio ke kiri/kanan"""
        _, sig_len = sig.shape
        shift_amt = int(np.random.random() * shift_limit * sig_len)
        return np.roll(sig, shift_amt)

    @staticmethod
    def add_noise(sig, noise_level=0.005):
        """Augmentasi: Menambah noise (kresek-kresek halus)"""
        noise = np.random.randn(sig.shape[0], sig.shape[1])
        augmented_sig = sig + noise * noise_level
        return augmented_sig