import torch
import torch.nn.functional as F

distributions = {
    'normal': torch.distributions.Normal,
    'tanh': torch.distributions.Normal,
    'beta': torch.distributions.Beta
}


class Agent:
    """Simple action-sampler class for an on-policy training algorithm

    Supports following policy distributions: Normal, tanh(Normal), Beta.
    For given game state produces differentiable action, log_prob_for_action, policy_entropy,
    which can be stored elsewhere for further loss computation
    """
    def __init__(self, net, device, distribution):
        self.net = net
        self.device = device
        self.distribution = distribution
        self.policy = distributions[distribution]

    def init_print(self):
        print('agent initialized with {} policy'.format(self.distribution))

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        param1, param2 = self.net(state)
        if self.distribution == 'beta':  # beta distribution requires parameters to be greater than 1
            param1, param2 = 1.0 + F.softplus(param1), 1.0 + F.softplus(param2)
        policy_distribution = self.policy(param1, param2)
        action = policy_distribution.sample()  # distribution parameters are batched -> samples are batched
        log_p_action = policy_distribution.log_prob(action)
        entropy = policy_distribution.entropy()
        if self.distribution == 'tanh':
            action = torch.tanh(action)
            log_p_action = log_p_action - \
                           torch.log(torch.tensor(4.0, dtype=torch.float32)) + \
                           2 * torch.log(torch.exp(action) + torch.exp(-action))
            # I guess entropy does not change for tanh(Normal)
        return action, log_p_action, entropy
