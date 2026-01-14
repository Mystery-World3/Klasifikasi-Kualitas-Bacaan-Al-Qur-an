# check_model.py
import torch
from src.model import ContrastiveModel, ClassifierModel
from src.config import Config

def test_contrastive_model():
    print("--- Cek Model Pre-training (Contrastive) ---")
    # Create dummy data : Batch 4, Channel 1, Panjang 48000
    dummy_input = torch.randn(4, 1, Config.N_SAMPLES)
    
    model = ContrastiveModel()
    
    # Enter data into the model
    features, projections = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Features (untuk classifier nanti): {features.shape} (Harus [4, 512])")
    print(f"Output Projection (untuk loss function): {projections.shape} (Harus [4, 128])")
    print("Sukses!\n")

def test_classifier_model():
    print("--- Cek Model Fine-tuning (Classifier) ---")
    dummy_input = torch.randn(4, 1, Config.N_SAMPLES)
    
    model = ClassifierModel(num_classes=5)
    
    output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Logits (Prediksi): {output.shape} (Harus [4, 5])")
    print("Sukses!\n")

if __name__ == "__main__":
    test_contrastive_model()
    test_classifier_model()