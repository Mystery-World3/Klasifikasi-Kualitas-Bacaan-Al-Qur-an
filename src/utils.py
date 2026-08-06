import random
import librosa
import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config

class AudioUtil:

    # OPEN AUDIO
    @staticmethod
    def open_audio(audio_path, sample_rate=None):

        if sample_rate is None:
            sample_rate = Config.SAMPLE_RATE

        try:
            sig, sr = librosa.load(
                audio_path,
                sr=sample_rate,
                mono=True
            )

            return sig

        except Exception as e:

            print(f"Error loading {audio_path}")
            print(e)
            return None

    # RECHANNEL
    @staticmethod
    def rechannel(sig):
        return sig

    # PAD / TRUNCATE
    @staticmethod
    def pad_trunc(sig, max_len):

        if sig is None:
            return np.zeros(max_len)
        if len(sig) > max_len:
            sig = sig[:max_len]
        elif len(sig) < max_len:
            pad = max_len - len(sig)
            sig = np.pad(
                sig,
                (0, pad),
                mode="constant"
            )

        return sig

    # TIME SHIFT
    @staticmethod
    def time_shift(sig):

        shift = int(
            random.uniform(-0.2,0.2) * len(sig)
        )

        sig = np.roll(sig, shift)

        return sig

    # ADD NOISE
    @staticmethod
    def add_noise(sig):
        noise = np.random.randn(len(sig))
        sig = sig + 0.005 * noise
        return sig

    # MEL SPECTROGRAM
    @staticmethod
    def audio_to_melspectrogram(sig):

        mel = librosa.feature.melspectrogram(

            y=sig,
            sr=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS

        )

        mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        mel = (

            mel - mel.min()

        ) / (

            mel.max() - mel.min() + 1e-8

        )

        return mel

    # FULL PREPROCESS
    @staticmethod
    def preprocess(
        audio_path,
        add_noise=False,
        shift=True
    ):

        sig = AudioUtil.open_audio(audio_path)
        if sig is None:
            return None
        sig = AudioUtil.rechannel(sig)
        sig = AudioUtil.pad_trunc(
            sig,
            Config.N_SAMPLES
        )

        if shift:
            sig = AudioUtil.time_shift(sig)

        if add_noise:
            sig = AudioUtil.add_noise(sig)

        # Convert ke Mel Spectrogram
        mel = AudioUtil.audio_to_melspectrogram(sig)

        tensor = torch.tensor(
            mel,
            dtype=torch.float32
        ).unsqueeze(0)

        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return tensor