import torch.nn as nn
from torch.nn.functional import selu


class Net(nn.Module):
    def __init__(self, num_frames):
        super(Net, self).__init__()
        num_actions = 5
        self.conv = nn.Sequential(
            nn.Conv2d(num_frames, 32, kernel_size=8, stride=4), nn.SELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.SELU(),
        )
        self.projector = nn.Linear(1344, 512)
        self.policy_stream = nn.Linear(512, 512)
        self.policy_layer = nn.Linear(512, num_actions)
        self.value_stream = nn.Linear(512, 512)
        self.value_layer = nn.Linear(512, 1)

    def forward(self, screen):
        batch = screen.size(0)
        conv_features = self.conv(screen)  # [batch, 64, 7, 3]
        features = selu(self.projector(conv_features.view(batch, -1)))

        policy_stream = selu(self.policy_stream(features))
        policy = self.policy_layer(policy_stream)

        value_stream = selu(self.value_stream(features))
        value = self.value_layer(value_stream).squeeze(-1)
        return policy, value
