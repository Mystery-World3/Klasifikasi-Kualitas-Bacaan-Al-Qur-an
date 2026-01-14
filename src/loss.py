# src/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    Ini adalah "Jantung" dari Contrastive Learning (SimCLR).
    """
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i: Proyeksi dari View 1 (Batch, 128)
        z_j: Proyeksi dari View 2 (Batch, 128)
        """
        batch_size = z_i.shape[0]
        
        # Gabungkan kedua view
        # Representations: [2*Batch, 128]
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Hitung kemiripan (Cosine Similarity) antar semua sampel
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        # SimCLR logic (Masking diagonal sendiri)
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        
        # Denominator: Sum semua exp similarity kecuali diri sendiri
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
        denominator = mask * torch.exp(similarity_matrix / self.temperature)
        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss