import torch
import torch.nn as nn
from torchvision import models


class MultimodalResNet18(nn.Module):
    def __init__(self, metadata_dim, hidden_dim=128, dropout=0.3):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        image_feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_encoder = backbone

        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image, metadata):
        image_features = self.image_encoder(image)
        metadata_features = self.metadata_encoder(metadata)
        fused = torch.cat([image_features, metadata_features], dim=1)
        out = self.classifier(fused)
        return out