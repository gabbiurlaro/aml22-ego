from torch import nn
import torch


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
    def forward(self, x):
        return self.classifier(x), {}


class MLP_late_fusion(nn.Module):
    def __init__(self, num_input, num_classes, num_clips) -> None:
        super().__init__()
        self.num_clips = num_clips
        self.num_input = num_input
        self.classifier = nn.Sequential(
            [
            nn.Linear(self.num_input, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
             ]
        )

    def forward(self, x):
        logits = []
        for clip in range(self.num_clips):
            logits.append(self.classifier(x[clip,:]))
        

        return torch.stack(logits, dim=0).mean(dim=0), {}
