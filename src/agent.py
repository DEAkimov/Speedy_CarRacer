import torch
import torch.nn.functional as F


distributions = {
    'normal': torch.distributions.Normal,
    'tanh': torch.distributions.Normal,
    'beta': torch.distributions.Beta
}


class Agent:
    def __init__(self, net, device):
        self.net = net
        self.device = device
        self.distribution = 'categorical'
        self.policy_distribution = torch.distributions.Categorical
        self.init_print()

    def init_print(self):
        print('agent initialized with {} policy'.format(self.distribution))

    def reset_noise(self):
        self.net.reset_noise()

    def policy(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        logits, value = self.net(state)
        policy_distribution = self.policy_distribution(logits=logits)
        return policy_distribution, value

    def log_p_for_action(self, state, action):
        policy, value = self.policy(state)
        log_p_for_action = policy.log_prob(action)
        entropy = policy.entropy()
        return log_p_for_action, value, entropy

    @staticmethod
    def tanh_correction(action, log_p_for_action):
        log_p_for_action = log_p_for_action - \
                           torch.log(torch.tensor(4.0, dtype=torch.float32)) + \
                           2 * torch.log(torch.exp(action) + torch.exp(-action))
        return log_p_for_action

    def act(self, state, greedy=False):
        policy, _ = self.policy(state)
        if greedy:
            # is it good for beta policy? I'm not sure
            action = policy.logits.argmax()
        else:
            action = policy.sample()
        if self.distribution == 'tanh':
            action = torch.tanh(action)
        return action
