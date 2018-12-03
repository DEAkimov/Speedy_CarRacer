import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, train_second_param):
        super(Net, self).__init__()
        action_dim = 3
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(),
        )
        self.projector = nn.Linear(1344, 512)
        # both Normal and Beta distributions have 2 parameters
        self.policy_param_1 = nn.Linear(512, action_dim)
        self.train_second = train_second_param
        if train_second_param:
            self.policy_param_2 = nn.Linear(512, action_dim)
        else:
            ones = torch.ones(1, action_dim, dtype=torch.float32)
            self.policy_param_2 = nn.Parameter(torch.log(0.1 * ones))
        self.value_layer = nn.Linear(512, 1)

    def forward(self, screen):
        batch = screen.size(0)
        conv_features = self.conv(screen)  # [batch, 64, 7, 3]
        features = self.projector(conv_features.view(batch, -1))
        param1 = self.policy_param_1(features)
        if self.train_second:
            param2 = self.policy_param_2(features)
        else:
            param2 = self.policy_param_2.expand_as(param1).detach()
        value = self.value_layer(features).squeeze(-1)
        return param1, param2, value
