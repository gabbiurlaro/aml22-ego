
from torch import nn
import torch
from utils.logger import logger
import numpy as np

class EMG_classifier(nn.Module):
    def __init__(self, num_input, num_classes, num_clips) -> None:
        super().__init__()
       
        self.num_input = num_input
        self.num_classes = num_classes
        self.num_clips = num_clips
        self.classifier = nn.Sequential( #16x32x32
            nn.Conv2d(num_input, 32, kernel_size=4, stride=1, padding=0),  # Output size: 32x29x29
            nn.BatchNorm2d(32),  # Apply batch normalization
            nn.ReLU(),  # Apply ReLU activation
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),  # Output size: 64x26x26
            nn.BatchNorm2d(64),  # Apply batch normalization
            nn.ReLU(),  # Apply ReLU activation
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0),  # Output size: 128x23x23
            nn.BatchNorm2d(128),  # Apply batch normalization
            nn.ReLU(),  # Apply ReLU activation
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),  # Output size: 256x20x20
            nn.BatchNorm2d(256),  # Apply batch normalization
            #nn.ReLU(),  # Apply ReLU activation
            #nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),  # Output size: 1024x20x20
            #nn.BatchNorm2d(1024),  # Apply batch normalization
            nn.ReLU(),  # Apply ReLU activation
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
            nn.Linear(256, 128),
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


class BasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels, pooling = True, activation = "relu", batchNorm = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 3, 1, 1)
        self.activation = nn.ReLU() if activation == "relu" else nn.Softmax()
        self.batchNorm = nn.BatchNorm2d(output_channels) if batchNorm else None
        self.pool = nn.MaxPool2d((2, 2))
        self.pooling = pooling

    def forward(self, x):
        x = self.conv(x)
        if self.batchNorm:
            x = self.batchNorm(x)
        x = self.activation(x)
        if self.pooling:
            x = self.pool(x)
        return x
    
class EMG_classifier_parametric(nn.Module):
    def __init__(self, input_size, output_size, num_classes, num_clips, **kwargs) -> None:
        super().__init__()
       
        self.input_size = input_size
        self.output_size = output_size
        self.num_classes = num_classes
        self.num_clips = num_clips

        self.use_batch_norm = kwargs.get("use_batch_norm", False)

        in_channels, in_height, in_width = input_size
        out_channels, out_height, out_width = output_size
        self.backbone = nn.Sequential()
        
        num_doubling_channels = int(np.log2(out_channels // in_channels))
        num_reducing_size_layers = int(np.log2(in_height / out_height))
        
        num_conv_pool_layers = min(num_doubling_channels, num_reducing_size_layers)

        current_channels = in_channels
        current_height = in_height
        for i in range(num_conv_pool_layers):
            self.backbone.append(BasicBlock(current_channels, current_channels * 2, pooling=True, batchNorm=self.use_batch_norm))
            current_channels = current_channels * 2
            current_height = current_height // 2
        
        if current_channels != out_channels:
            # we need only to adjust the channel size
            for i in range(num_doubling_channels - num_conv_pool_layers):
                self.backbone.append(BasicBlock(current_channels, current_channels * 2, pooling=False, batchNorm=self.use_batch_norm))
                current_channels = current_channels * 2

        if current_height != out_height:
            # we need only to adjust the height size ad width size
            for i in range(num_reducing_size_layers - num_conv_pool_layers):
                self.backbone.append(BasicBlock(current_channels, current_channels, pooling=True, batchNorm=self.use_batch_norm))
                current_height = current_height // 2

        fc_input_size = out_channels * out_height * out_width
        fc_hidden_size = (out_channels * out_height * out_width) // 2
        
        self.on_top_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, fc_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_size, num_classes),
            nn.Dropout(p=0.6)
        )

    def forward(self, x):
        logits = []
        feats = []
        for clip in range(self.num_clips):
            _, channels, height, width = x[clip].shape
            
            if channels != self.input_size[0] or height != self.input_size[1] or width != self.input_size[2]:
                raise ValueError(f"Input shape {x[clip].shape} does not match the expected shape {self.input_size}")
            y = self.backbone(x[clip]).squeeze()

            feats.append(y)
            logit = self.on_top_classifier(y)
            logits.append(logit)
        return torch.stack(logits, dim=0).mean(dim=0), {i: feats[i] for i in range(self.num_clips)}
