# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models

class ContrastiveModel(nn.Module):
    def __init__(self, num_classes=5, mode='finetune'):
        super(ContrastiveModel, self).__init__()
        self.mode = mode
        
        # 1. Encoder Utama (ResNet-18)
        self.backbone = models.resnet18(pretrained=True)
        
        # Ubah input channel jadi 1 (karena Spectrogram hitam putih)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Ambil jumlah fitur sebelum layer terakhir (biasanya 512)
        num_features = self.backbone.fc.in_features
        
        # Hapus FC asli
        self.backbone.fc = nn.Identity()

        # 2. Projection Head (Dipakai saat Stage 1: Pre-train)
        # Tugas: Memproyeksikan fitur ke ruang dimensi lain untuk dihitung loss-nya
        self.projection_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # 3. Classifier Head (Dipakai saat Stage 2: Fine-tune/Skripsi)
        # Tugas: Menentukan Mumtaz, Jayyid, dll.
        self.classifier_head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Ekstrak fitur dari gambar audio
        features = self.backbone(x)
        
        if self.mode == 'pretrain':
            # Kembalikan hasil proyeksi (vektor)
            return self.projection_head(features)
        else:
            # Kembalikan hasil prediksi kelas (logits)
            return self.classifier_head(features)