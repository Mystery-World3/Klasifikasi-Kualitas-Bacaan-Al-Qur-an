# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from src.config import Config
from src.dataset import ContrastivePretrainDataset, FineTuneDataset
from src.model import ContrastiveModel, ClassifierModel
from src.loss import NTXentLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Menggunakan device: {device}")

def train_contrastive_phase():
    print("\n=== MEMULAI FASE 1: CONTRASTIVE PRE-TRAINING (UNLABELED) ===")
    
    # 1. prepare Data
    dataset = ContrastivePretrainDataset(Config.UNLABELED_DIR)
    if len(dataset) == 0:
        print("Error: Data Unlabeled kosong! Fase 1 dibatalkan.")
        return None
        
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # 2. prepare Model & Loss
    model = ContrastiveModel().to(device)
    criterion = NTXentLoss(temperature=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Learning rate standar
    
    # 3. Training Loop
    EPOCHS = 5  
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (view1, view2) in enumerate(loader):
            view1, view2 = view1.to(device), view2.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            _, proj1 = model(view1) 
            _, proj2 = model(view2)
            
            # calculate Loss
            loss = criterion(proj1, proj2)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")
        
    # 4. save encoder (Otak yang sudah pintar fitur)
    os.makedirs("models", exist_ok=True)
    torch.save(model.encoder.state_dict(), "models/encoder_pretrained.pth")
    print(">>> Pre-training Selesai! Model disimpan di models/encoder_pretrained.pth")
    return model.encoder

def train_finetune_phase(pretrained_encoder=None):
    print("\n=== MEMULAI FASE 2: SUPERVISED FINE-TUNING (LABELED) ===")
    
    # 1. prepare Data
    dataset = FineTuneDataset(Config.LABELED_DIR)
    if len(dataset) == 0:
        print("Error: Data Labeled kosong! Fase 2 dibatalkan.")
        return

    # Split dataset (80% Train, 20% Val) - Sederhana manual
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # 2. prepare Model Classifier
    # load the encoder that was trained in Phase 1
    model = ClassifierModel(encoder=pretrained_encoder, num_classes=5).to(device)
    
    # OPSI: Bekukan encoder (Freeze) agar tidak rusak fiturnya, atau biarkan ikut belajar (Fine-tune).
    # Untuk data sedikit, biasanya lebih aman di-Freeze dulu atau learning rate kecil.
    # Di sini kita biarkan terbuka tapi LR kecil.
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    
    # 3. Training Loop
    EPOCHS = 10 
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for audio, labels in train_loader:
            audio, labels = audio.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(audio) # Output: Logits (5 kelas)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # calculate the rough training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        
    # 4. save the Final Model
    torch.save(model.state_dict(), "models/final_model_skripsi.pth")
    print(">>> Fine-tuning Selesai! Model akhir disimpan di models/final_model_skripsi.pth")

if __name__ == "__main__":
    # run Phase 1
    trained_encoder = train_contrastive_phase()
    
    # run Phase 2 
    
    train_finetune_phase(pretrained_encoder=trained_encoder)