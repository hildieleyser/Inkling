# ssvep/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, n_chans=8, n_samples=500, n_classes=12, dropout=0.3):
        super().__init__()

        self.conv_time = nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        self.depthwise = nn.Conv2d(
            8, 16, kernel_size=(n_chans, 1), groups=8, bias=False
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8),
                      groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False)
        )
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        dummy = torch.zeros(1, 1, n_chans, n_samples)
        with torch.no_grad():
            feat_dim = self._features(dummy).shape[1]

        self.classifier = nn.Linear(feat_dim, n_classes)

    def _features(self, x):
        x = F.elu(self.bn1(self.conv_time(x)))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = self.drop1(self.pool1(x))
        x = F.elu(self.bn3(self.separable(x)))
        x = self.drop2(self.pool2(x))
        return x.flatten(start_dim=1)

    def forward(self, x):
        return self.classifier(self._features(x))
