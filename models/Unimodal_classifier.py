nn.Linear(fc_input_size, fc_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_size, num_classes),
            nn.Dropout(self.dropout_rate)


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