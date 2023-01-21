from models import RelationModuleMultiScale

import torch
from torch.autograd import Variable

if __name__ == "__main__":
    batch_size = 32
    num_frames = 5
    num_class = 8
    img_feature_dim = 1024
    print(f"Feeding the TRN with a batch of dimension {batch_size}, with {num_frames} and a feature dimension of {img_feature_dim}")
    input_var = Variable(torch.randn(batch_size, num_frames, img_feature_dim))
    model = RelationModuleMultiScale(img_feature_dim, num_frames, num_class)
    output = model(input_var)
    print(f"output: {output}\n\nSize: {output.size()}")

    print(f"Prediction: {torch.argmax(output, axis=1)}")
