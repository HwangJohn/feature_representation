import torch
import torch.nn as nn
from torchinfo import summary

class SEModule(nn.Module):
    """ SequeezeExcitationModule
        input: [B, C, H, W] torch tensor
        output: [B, C, H, W] torch tensor
    """
    def __init__(self, in_channel):
        super().__init__()
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.down_linear = nn.Linear(in_channel, in_channel//2)
        self.up_linear = nn.Linear(in_channel//2, in_channel)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        # main branch
        skip_connection = x

        # squeeze and excitation branch
        x = self.global_avg(x)
        x = self.flatten(x)
        x = self.down_linear(x)
        x = self.relu(x)
        x = self.up_linear(x)
        x = self.sigmoid(x)
        x = x.reshape(b, c, 1, 1)

        # Channel-wise recalibration
        x = x * skip_connection
    
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    se_module = SEModule(8).to(device)
    test_input = torch.rand((1,8,10,10))
    output = se_module(test_input)
    print(output.shape)
    summary(se_module, (1, 8, 10, 10))