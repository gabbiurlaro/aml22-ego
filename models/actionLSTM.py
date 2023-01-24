import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionLSTM(nn.Module):
    def __init__(self, feature_dim, num_classes, num_clips=5) -> None:
        super(ActionLSTM, self).__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_clips = num_clips

        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=512, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        #h_0 = ( torch.randn(1, len(x), self.num_classes), torch.randn(1, len(x), self.num_classes))
        out, hidden_state = self.lstm(x.permute([1, 0, 2]))
        # print(f"out: {out.size()}, h: {hidden_state[0].size()}")
        out = self.classifier(hidden_state[-1])
        # print(f"Classifier out is: {out.size()}")
        return out.squeeze(), {}