# check_model.py
import torch
from src.config import Config
from src.model import ClassifierModel

def test_classifier_model():
    print("--- Cek Arsitektur Model (ResNet-18 Modified) ---")
    
    # 1. Inisialisasi Model
    try:
        model = ClassifierModel(num_classes=5)
        print("✅ Model berhasil diinisialisasi.")
    except Exception as e:
        print(f"❌ Gagal init model: {e}")
        return

    # 2. Buat Dummy Input (Pura-pura ini Spectrogram)
    # Bentuk: (Batch Size, Channel, Tinggi/Freq, Lebar/Waktu)
    # ResNet butuh 4D Tensor.
    batch_size = 2
    channels = 1 # Audio spectrogram cuma 1 channel (hitam putih)
    height = Config.N_MELS # Contoh: 128
    width = 200 # Panjang waktu (time steps)
    
    dummy_input = torch.randn(batch_size, channels, height, width)
    print(f"📉 Input Dummy Shape: {dummy_input.shape} (Batch, Ch, Mel, Time)")
    
    # 3. Masukkan ke Model (Forward Pass)
    try:
        output = model(dummy_input)
        print(f"📈 Output Logits Shape: {output.shape}")
        
        # Cek apakah outputnya sesuai (Batch Size, 5 Kelas)
        expected_shape = (batch_size, 5)
        if output.shape == expected_shape:
            print(f"✅ Tes Sukses! Output shape sesuai {expected_shape}")
        else:
            print(f"❌ Tes Gagal! Output shape tidak sesuai.")
            
    except Exception as e:
        print(f"❌ Error saat Forward Pass: {e}")
        print("Tips: Cek apakah layer awal ResNet sudah diubah jadi 1 channel?")

if __name__ == "__main__":
    test_classifier_model()