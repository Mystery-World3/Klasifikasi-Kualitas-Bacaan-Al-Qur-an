import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

file_path = "data/labeled/mumtaz/P001_01.wav"

# Cek dulu filenya ada atau nggak
if not os.path.exists(file_path):
    print(f"❌ Yah, file {file_path} nggak ketemu. Cek lagi foldernya Bro!")
else:
    print(f"⏳ Memproses file: {file_path}...")
    
    # 2. Load audio
    y, sr = librosa.load(file_path, sr=16000)

    # 3. Hitung Mel-Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 4. Menggambar Plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Visualisasi Mel-Spectrogram')
    plt.tight_layout()

    # 5. Save Gambar
    output_nama = "melspectrogram.png"
    plt.savefig(output_nama, dpi=300, bbox_inches='tight')
    
    print(f"✅ Gambar berhasil di-save dengan nama: {output_nama}")