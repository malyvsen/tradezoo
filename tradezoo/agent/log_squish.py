import torch


class LogSquish(torch.nn.Module):
    """An activation function which never saturates"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sign(x) * torch.log(1 + torch.abs(x))
