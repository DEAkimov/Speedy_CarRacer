import torch.nn as nn
from torch.nn.functional import selu
from .noisy_linear_layer import NoisyLinear


class Net(nn.Module):
    def __init__(self, num_frames, noisy):
        super(Net, self).__init__()
        num_actions = 5
        self.noisy = noisy
        if noisy:
            linear = NoisyLinear
        else:
            linear = nn.Linear
        self.conv = nn.Sequential(
            nn.Conv2d(num_frames, 16, kernel_size=8, stride=4), nn.SELU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3), nn.SELU(),
        )
        self.projector = linear(672, 256)
        self.policy_layer = linear(256, num_actions)
        self.value_layer = linear(256, 1)
        self.init_print()

    def init_print(self):
        number_of_parameters = sum(p.numel() for p in self.parameters())
        print('network initialized. Noisy: {}, number of parameters: {}'.format(
            self.noisy, number_of_parameters
        ))
        print(self)

    def reset_noise(self):
        if self.noisy:
            for layer in [self.projector,
                          self.policy_stream, self.policy_layer,
                          self.value_stream, self.value_layer]:
                layer.reset_noise()

    def forward(self, screen):
        batch = screen.size(0)
        conv_features = self.conv(screen)  # [batch, 64, 7, 3]
        features = selu(self.projector(conv_features.view(batch, -1)))
        policy = self.policy_layer(features)
        value = self.value_layer(features).squeeze(-1)
        return policy, value
