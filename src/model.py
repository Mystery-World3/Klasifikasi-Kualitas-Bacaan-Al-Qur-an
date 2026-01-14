# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models
from .config import Config

class ClassifierModel(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(ClassifierModel, self).__init__()
        
        # --- Transfer Learning ---
        # Menggunakan ResNet18 yang sudah dilatih (pretrained) pada ImageNet.
        # Referensi Arsitektur: He et al., "Deep Residual Learning for Image Recognition"
        self.backbone = models.resnet18(pretrained=True)
        
        # --- Modifikasi Layer Input ---
        # ResNet asli menerima gambar RGB (3 channel).
        # Spectrogram audio kita hanya 1 channel (Grayscale/Hitam Putih).
        # Kita ubah layer pertama (conv1) agar menerima in_channels=1.
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # --- Modifikasi Layer Output (Classification Head) ---
        # Mengganti Fully Connected Layer terakhir agar outputnya sesuai jumlah kelas kita (5).
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5), # Regularisasi untuk mencegah overfitting
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        # Forward pass: Input -> ResNet Backbone -> Logits
        return self.backbone(x)