import torch.nn as nn
import torch
import torch.nn.functional as F


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


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
