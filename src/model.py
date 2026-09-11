import torch
import torch.nn as nn
import torchvision.models as models


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        num_classes=5,
        mode="finetune",
        pretrained=True
    ):
        super(ContrastiveModel, self).__init__()

        self.mode = mode

        # Untuk training:
        # pretrained=True -> gunakan bobot ImageNet ResNet18
        #
        # Untuk inference/deployment:
        # pretrained=False -> tidak download bobot dari internet
        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        # 1. Backbone ResNet18
        self.backbone = models.resnet18(
            weights=weights
        )

        # Input spectrogram adalah 1 channel
        self.backbone.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Jumlah fitur sebelum FC
        num_features = self.backbone.fc.in_features

        # Hilangkan FC bawaan ResNet
        self.backbone.fc = nn.Identity()

        # 2. Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # 3. Classification Head
        self.classifier_head = nn.Linear(
            num_features,
            num_classes
        )

    def forward(self, x):
        # Ekstraksi fitur
        features = self.backbone(x)

        # Stage 1 - contrastive learning
        if self.mode == "pretrain":
            return self.projection_head(features)

        # Stage 2 - classification
        return self.classifier_head(features)