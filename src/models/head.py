import torch
import torch.nn as nn

class CombineModel(nn.Module):
    def __init__(self, siglip2_dim, dino_dim, num_s2_classes):
        super().__init__()
        #combine both vectors into one vector of size siglip_dim + dino_dim, then pass through nn to get final prediction of s2 cell class
        fusion_dim = siglip2_dim + dino_dim
        
        #use dropout to prevent overfitting, GELU for nonlinearity
        self.network = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, num_s2_classes)
        )

    def forward(self, siglip_features, dino_features):
        fused = torch.cat((siglip_features, dino_features), dim=1)
        return self.network(fused)