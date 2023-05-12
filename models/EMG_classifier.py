
from torch import nn
import torch

class EMG_classifier(nn.Module):
    def __init__(self, num_input, num_classes) -> None:
        super().__init__()
        self.num_input = num_input
        self.classifier = nn.Sequential(
            nn.Conv2d(num_input, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.MaxPool2d(2),
            nn.Linear(256, 128),
            nn.Linear(128, 11)
        )

    def forward(self, x):
        logits = []
        for clip in range(self.num_clips):
            logits.append(self.classifier(x[clip,:]))

        return torch.stack(logits, dim=0).mean(dim=0), {}
