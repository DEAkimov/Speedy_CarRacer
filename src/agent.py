import torch


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

    def act(self, state, greedy=False):
        with torch.no_grad():
            policy, value = self.policy(state)
        if greedy:
            action = policy.logits.argmax(-1)
        else:
            action = policy.sample()
        return action

    def __call__(self, state):
        policy, value = self.policy(state)
        action = policy.sample()
        log_p_for_action = policy.log_prob(action)
        entropy = policy.entropy()
        return action, value, log_p_for_action, entropy
