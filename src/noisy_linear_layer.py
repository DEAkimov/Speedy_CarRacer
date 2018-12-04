import torch
import torch.nn as nn
from torch.nn.functional import linear
import math


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = 0.017

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()
        self.epsilon_w, self.epsilon_b = None, None
        self.reset_noise()

    def reset_parameters(self):
        mu_range = math.sqrt(3.0 / self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init)

    # noinspection PyUnresolvedReferences
    def reset_noise(self):
        self.epsilon_w = torch.randn(self.out_features, self.in_features)
        self.epsilon_b = torch.randn(self.out_features)

    def forward(self, x):
        w = self.weight_mu + self.weight_sigma * self.epsilon_w
        b = self.bias_mu + self.bias_sigma * self.epsilon_b
        return linear(x, w, b)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
