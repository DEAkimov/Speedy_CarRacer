import torch.nn as nn
from torch.nn.functional import selu


class Net(nn.Module):
    def __init__(self, num_frames):
        super(Net, self).__init__()
        num_actions = 5
        self.conv = nn.Sequential(
            nn.Conv2d(num_frames, 16, kernel_size=8, stride=4, bias=False), nn.SELU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, bias=False), nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3, bias=False), nn.SELU(),
            nn.Conv2d(32, 64, kernel_size=3, bias=False), nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3, bias=False), nn.SELU(),
        )
        self.policy_projector = nn.Linear(64 * 3 * 3, 96, bias=False)
        self.policy_layer = nn.Linear(96, num_actions, bias=False)

        self.value_projector = nn.Linear(64 * 3 * 3, 96, bias=False)
        self.value_layer = nn.Linear(96, 1, bias=False)

        self.init_print()

    def init_print(self):
        number_of_parameters = sum(p.numel() for p in self.parameters())
        print('network initialized. Number of parameters: {}'.format(number_of_parameters))
        print(self)

    def forward(self, screen):
        batch = screen.size(0)
        conv_features = self.conv(screen).view(batch, -1)
        policy = selu(self.policy_projector(conv_features))
        policy = self.policy_layer(policy)

        value = selu(self.value_projector(conv_features))
        value = self.value_layer(value).squeeze(-1)
        return policy, value
