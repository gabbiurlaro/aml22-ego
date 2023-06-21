
from torch import nn
import torch
from utils.logger import logger
import numpy as np

class BasicBlock(nn.Module):
    """Basic block for the EMG classifier.
    It consist of a convolutional layer, an activation function and an optional pooling layer.
    It allow to modify the dimensionality of the feature map.
    """
    def __init__(self, input_channels, output_channels, **kwargs) -> None:
        super().__init__()

        activation = kwargs.get("activation", "relu")
        batchNorm = kwargs.get("batchNorm", False)
        self.pooling = kwargs.get("pooling", True)

        self.conv = nn.Conv2d(input_channels, output_channels, 3, 1, 1)
        self.activation = nn.ReLU() if activation == "relu" else nn.Softmax()
        self.batchNorm = nn.BatchNorm2d(output_channels) if batchNorm else None
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.conv(x)
        if self.batchNorm:
            x = self.batchNorm(x)
        x = self.activation(x)
        if self.pooling:
            x = self.pool(x)
        return x
    
class EMG_classifier_parametric(nn.Module):
    """
    EMG classifier model for action recognition.
    Based on a convolutional neural network and a fully connected classifier.
    """
    def __init__(self, input_size, output_size, num_classes, num_clips, **kwargs) -> None:
        super().__init__()
       
        self.input_size = input_size
        self.output_size = output_size
        self.num_classes = num_classes
        self.num_clips = num_clips
        
        self.include_top = kwargs.get("include_top", True)
        self.use_batch_norm = kwargs.get("use_batch_norm", False)
        self.dropout_rate = kwargs.get("dropout_rate", 0.6)

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
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        logits = []
        feats = []
        for clip in range(self.num_clips):
            _, channels, height, width = x[clip].shape
            
            if channels != self.input_size[0] or height != self.input_size[1] or width != self.input_size[2]:
                raise ValueError(f"Input shape {x[clip].shape} does not match the expected shape {self.input_size}")
            y = self.backbone(x[clip])
            feats.append(y.squeeze())
            if self.include_top:
                logit = self.on_top_classifier(y)
                logits.append(logit)
        if self.include_top:
            return torch.stack(logits, dim=0).mean(dim=0), {i: feats[i] for i in range(self.num_clips)}
        else:
            {i: feats[i] for i in range(self.num_clips)}