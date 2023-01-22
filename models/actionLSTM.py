import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionLSTM(nn.Module):
    def __init__(self, feature_dim, num_classes, num_clips=5) -> None:
        super(ActionLSTM, self).__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_clips = num_clips

        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=self.num_classes, num_layers=1)

    def forward(self, x):
        #h_0 = ( torch.randn(1, len(x), self.num_classes), torch.randn(1, len(x), self.num_classes))
        
        _, hidden_state = self.lstm(x.view(self.num_clips, len(x), self.feature_dim))
        return hidden_state[0].reshape(len(x), self.num_classes), {}