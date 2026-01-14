# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEncoder(nn.Module):
    """
    Backbone Utama: Mengubah Audio Mentah (1, 48000) menjadi Vektor Fitur (128).
    Menggunakan 1D Convolution.
    """
    def __init__(self):
        super(AudioEncoder, self).__init__()
        
        # Layer 1: Stride besar untuk mengurangi dimensi audio yang panjang
        # Input: [Batch, 1, 48000]
        self.conv1 = nn.Conv1d(1, 128, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)

        # Layer 2
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)

        # Layer 3
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)

        # Layer 4
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)

        # Global Average Pooling (Meratakan fitur waktu menjadi satu vektor)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Output Features dimension
        self.feature_dim = 512

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)

        x = self.gap(x) # [Batch, 512, 1]
        x = x.squeeze(-1) # [Batch, 512]
        return x

class ContrastiveModel(nn.Module):
    """
    Model untuk Fase Pre-training (Unsupervised).
    Terdiri dari: Encoder + Projection Head.
    """
    def __init__(self):
        super(ContrastiveModel, self).__init__()
        self.encoder = AudioEncoder()
        
        # Projection Head: Mengubah fitur 512 menjadi 128 untuk loss function
        # Ini teknik standar SimCLR agar representasi lebih bagus.
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection(features)
        # Normalisasi vektor agar panjangnya = 1 (Penting untuk Contrastive Loss)
        projections = F.normalize(projections, dim=1)
        return features, projections

class ClassifierModel(nn.Module):
    """
    Model untuk Fase Fine-tuning (Supervised).
    Terdiri dari: Encoder (yang sudah dilatih) + Classification Head.
    """
    def __init__(self, encoder=None, num_classes=5):
        super(ClassifierModel, self).__init__()
        # Jika ada encoder pre-trained, pakai itu. Jika tidak, buat baru.
        self.encoder = encoder if encoder else AudioEncoder()
        
        # Classification Head: 512 -> 5 Kelas (Mumtaz s.d Rosib)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2), # Mencegah overfitting
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits