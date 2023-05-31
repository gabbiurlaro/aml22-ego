
from torch import nn
import torch
from utils.logger import logger

class EMG_classifier(nn.Module):
    def __init__(self, num_input, num_classes, num_clips) -> None:
        super().__init__()
       
        self.num_input = num_input
        self.num_classes = num_classes
        self.num_clips = num_clips
        self.emdedding_size = 1024
        self.classifier = nn.Sequential( 
            #16x32x32
            nn.Conv2d(num_input, 32, kernel_size=3, stride=2, padding=1),  # Output size: 32x16x16
            nn.BatchNorm2d(32),  # Apply batch normalization
            nn.ReLU(),  # Apply ReLU activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output size: 64x8x8
            nn.BatchNorm2d(64),  # Apply batch normalization
            nn.ReLU(),  # Apply ReLU activation
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output size: 128x4x4
            nn.BatchNorm2d(128),  # Apply batch normalization
            nn.ReLU(),  # Apply ReLU activation
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output size: 256x2x2
            nn.BatchNorm2d(256),  # Apply batch normalization
            nn.ReLU(),  # Apply ReLU activation
            nn.Conv2d(256,  self.emdedding_size , kernel_size=3, stride=2, padding=1),# Output size: 1024x1x1
            nn.BatchNorm2d( self.emdedding_size ),  # Apply batch normalization
            nn.Sigmoid(),  # Apply ReLU activation
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # Output size: 1024x1x1
            # nn.Conv2d(num_input, 32, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.MaxPool2d(2, stride=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.emdedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        logits = []
        feats = []
        for clip in range(self.num_clips):
            y = self.classifier(x[clip,:])
            feats.append(y.squeeze())
            #print(f'y shape: {y.shape}') 
            logits.append(self.fc(y))
        return torch.stack(logits, dim=0).mean(dim=0), {i: feats[i] for i in range(self.num_clips)}

