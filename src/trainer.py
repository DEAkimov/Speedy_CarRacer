import torch


class Trainer:
    def __init__(self,
                 train_environment, test_environment, num_tests,
                 agent, device, entropy_reg, optimizer,
                 writer):
        self.train_environment = train_environment
        self.test_environment = test_environment
        self.num_tests = num_tests

        self.agent = agent
        self.device = device
        self.entropy_reg = entropy_reg
        self.optimizer = optimizer

        self.writer = writer

        self.add = torch.tensor([0, 1, 1], dtype=torch.float32, device=device)
        self.mul = torch.tensor([1, 0.5, 0.5], dtype=torch.float32, device=device)

    def project_actions(self, actions):
        return (actions + self.add) * self.mul

    def collect_batch(self, time_steps, observation):
        observations, actions, rewards, is_done = [observation], [], [], []
        for step in range(time_steps):
            action, _ = self.agent.act(observation)
            env_action = self.project_actions(action).cpu().numpy()
            observation, reward, done, _ = self.train_environment.step(env_action)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            is_done.append(done)
        observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        actions = torch.stack(actions, dim=0)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        is_done = torch.tensor(is_done, dtype=torch.float32, device=self.device)
        return observations, actions, rewards, is_done
