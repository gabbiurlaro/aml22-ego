
from torch import nn
import torch
from utils.logger import logger
import numpy as np

class Unimodal_classifier_parametric(nn.Module):
    """
    EMG classifier model for action recognition.
    Based on a convolutional neural network and a fully connected classifier.
    """
    def __init__(self, input_size, num_classes, num_clips, **kwargs) -> None:
        super().__init__()
       
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_clips = num_clips
        
        self.dropout_rate = kwargs.get("dropout_rate", 0.6)

        self.classifier = nn.Sequential(nn.Linear(input_size, int(input_size/4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(input_size/4), num_classes),
            nn.Dropout(self.dropout_rate))
    
    def forward(self, x):
        logits = []
        for clip in range(self.num_clips):    
            logit = self.classifier(x[clip])
            logits.append(logit)
       
        return torch.stack(logits, dim=0).mean(dim=0), {}
