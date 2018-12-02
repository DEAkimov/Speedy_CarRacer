import torch
import torch.nn.functional as F


distributions = {
    'normal': torch.distributions.Normal,
    'tanh': torch.distributions.Normal,
    'beta': torch.distributions.Beta
}


class Agent:
    def __init__(self, net, device, distribution):
        self.net = net
        self.device = device
        self.distribution = distribution
        self.policy_distribution = distributions[distribution]

    def init_print(self):
        print('agent initialized with {} policy'.format(self.distribution))

    def policy(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        param1, param2, value = self.net(state)
        if self.distribution == 'beta':  # beta distribution requires parameters to be greater than 1
            param1, param2 = 1.0 + F.softplus(param1), 1.0 + F.softplus(param2)
        policy_distribution = self.policy_distribution(param1, param2)
        return policy_distribution, value

    def log_p_for_action(self, state, action):
        policy, value = self.policy(state)
        log_p_for_action = policy.log_prob(action)
        if self.distribution == 'tanh':
            # here action is assumed to be tanh(z)
            log_p_for_action = self.tanh_correction(action, log_p_for_action)
        return log_p_for_action, value

    @staticmethod
    def tanh_correction(action, log_p_for_action):
        log_p_for_action = log_p_for_action - \
                           torch.log(torch.tensor(4.0, dtype=torch.float32)) + \
                           2 * torch.log(torch.exp(action) + torch.exp(-action))
        return log_p_for_action

    def act(self, state):
        policy, value = self.policy(state)
        action = policy.sample()
        if self.distribution == 'tanh':
            action = torch.tanh(action)
        return action, value
