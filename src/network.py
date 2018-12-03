import torch.nn as nn
from torch.nn.functional import selu


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_actions = 5
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.SELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.SELU(),
        )
        self.projector = nn.Linear(1344, 512)
        self.policy_layer = nn.Linear(512, num_actions)
        self.value_layer = nn.Linear(512, 1)

    def forward(self, screen):
        batch = screen.size(0)
        conv_features = self.conv(screen)  # [batch, 64, 7, 3]
        features = selu(self.projector(conv_features.view(batch, -1)))
        policy = self.policy_layer(features)
        value = self.value_layer(features).squeeze(-1)
        return policy, value
