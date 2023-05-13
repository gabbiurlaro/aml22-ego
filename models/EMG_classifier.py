
from torch import nn
import torch

class EMG_classifier(nn.Module):
    def __init__(self, num_input, num_classes, num_clips) -> None:
        super().__init__()
        self.num_input = num_input
        self.num_classes = num_classes
        self.num_clips = num_clips
        self.classifier = nn.Sequential(
            nn.Conv2d(num_input, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(2, stride=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=num_classes)
        )

    def forward(self, x):
        logits = []
        for clip in range(self.num_clips):
            y = self.classifier(x[clip,:])
            y.squeeze_()
            print(f'y shape: {y.shape}') 
            logits.append(self.fc(y))

        return torch.stack(logits, dim=0).mean(dim=0), {}
