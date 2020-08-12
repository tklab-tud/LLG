import torch
import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):
    def __init__(self, parameter):
        super(Net1, self).__init__()
        act = nn.Sigmoid
        self.parameter = parameter

        self.body = nn.Sequential(
            nn.Conv2d(parameter["channel"], 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(parameter["hidden"], parameter["num_classes"])
        )

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())
