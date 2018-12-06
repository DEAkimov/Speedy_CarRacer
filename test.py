import torch
from src import Net, Agent, create_env, Trainer

if __name__ == '__main__':
    net = Net(5, False)
    checkpoint = torch.load('logs/a2c/default/epoch_1.pth')
    net.load_state_dict(checkpoint['net'])
    device = torch.device('cpu')
    agent = Agent(net, device)

    _, env = create_env(False, 1, 5, 5)
    t = Trainer(_, 0, env, 1, agent, device, None, 'mse',
                1e-3, 0.99, 0.95, False, False, 'logs/a2c/test/')
    reward = t.test_performance(True)
    print(reward)
