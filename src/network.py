import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        action_dim = 3
        self.conv = nn.Sequential(  # empirically produces tensor of shape [batch, 64, 6, 6]
            nn.Conv2d(3, 32, kernel_size=7, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU()
        )
        self.projector = nn.Linear(2304, 512)
        # both Normal and Beta distributions have 2 parameters
        self.policy_param_1 = nn.Linear(512, action_dim)
        self.policy_param_2 = nn.Linear(512, action_dim)
        self.value_layer = nn.Linear(512, 1)

    def forward(self, screen):
        batch = screen.size(0)
        conv_features = self.conv(screen)
        features = self.projector(conv_features.view(batch, -1))
        param1 = self.policy_param_1(features)
        param2 = self.policy_param_2(features)
        value = self.value_layer(features).squeeze(-1)
        return param1, param2, value
