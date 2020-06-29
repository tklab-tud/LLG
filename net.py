import torch.nn as nn


class Net(nn.Module):
    def __init__(self, parameter):
        super(Net, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(parameter["channel"], 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(588, parameter["num_classes"])
        )

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def weights_init(self):
        try:
            if hasattr(self, "weight"):
                self.weight.data.uniform_(-0.5, 0.5)
        except Exception:
            print('warning: failed in weights_init for %s.weight' % self._get_name())
        try:
            if hasattr(self, "bias"):
                self.bias.data.uniform_(-0.5, 0.5)
        except Exception:
            print('warning: failed in weights_init for %s.bias' % self._get_name())
