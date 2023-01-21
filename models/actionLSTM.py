import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionLSTM(nn.Module):
    def __init__(self, num_classes, feature_dim=1024) -> None:
        super(ActionLSTM, self).__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.lstm = nn.LSTM(self.feature_dim, self.num_classes)

    def forward(self, clips):
        _, hidden_state = self.lstm(clips.view(len(clips), 1, -1))
        return hidden_state