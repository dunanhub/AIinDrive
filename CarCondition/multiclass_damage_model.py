import os
import torch.nn as nn
from torchvision import models

class MulticlassDamageModel(nn.Module):
    def __init__(self, num_classes=3, dropout=0.6):
        super().__init__()
        # если нет интернета, можно отключить скачивание предобученных весов:
        use_pretrained = os.getenv("RESNET_WEIGHTS", "IMAGENET1K_V2").upper()
        if use_pretrained == "NONE":
            self.backbone = models.resnet50(weights=None)
        else:
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            )

        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 1024), nn.ReLU(inplace=True), nn.BatchNorm1d(1024),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        import torch
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x