import torch
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from .utils import save


class Trainer:
    def __init__(self,
                 train_environment, num_environments, test_environment, num_tests,
                 agent, device, optimizer, value_loss,
                 entropy_reg, gamma, gae_lambda, normalize_advantage,
                 use_gae, logdir):
        # environments
        self.train_environment = train_environment
        self.num_environments = num_environments
        self.test_environment = test_environment
        self.num_tests = num_tests
        self.last_observation = self.train_environment.reset()

        # agent and optimizer
        self.agent = agent
        self.device = device
        self.optimizer = optimizer
        assert value_loss in ['mse', 'huber'], \
            'loss should be \'mse\' or \'huber\', you provide \'{}\''.format(value_loss)
        if value_loss == 'huber':
            self.value_loss = torch.nn.SmoothL1Loss()
        elif value_loss == 'mse':
            self.value_loss = torch.nn.MSELoss()

        # training parameters
        self.entropy_reg = entropy_reg
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_gae = use_gae
        self.normalize_advantage = normalize_advantage

        # writer and statistics
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        self.worker_reward = np.zeros((num_environments,), dtype=np.float32)
        self.episodes_done = np.zeros((num_environments,), dtype=np.uint16)

        # action projector parameters
        # noinspection PyCallingNonCallable
        self.discrete_actions = torch.tensor([
            [-1, 0, 0],  # turn left
            [+1, 0, 0],  # turn right
            [0, 1, 0],  # accelerate
            [0, 0, 0.8],  # decelerate
            [0, 0, 0]  # do nothing
        ], dtype=torch.float32)
        # print trainer parameters
        self.init_print(value_loss)

    def init_print(self, vl):
        print('trainer initialized, logdir: {}'.format(self.logdir))
        print('training parameters:')
        print('\t value_loss: {}, gamma: {}, entropy_reg: {}'.format(vl, self.gamma, self.entropy_reg))
        print('\t use_gae: {}, gae_lambda: {}, normalize_adv: {}'.format(
            self.use_gae, self.gae_lambda, self.normalize_advantage
        ))

    def project_actions(self, actions):
        return self.discrete_actions[actions]

    def test_performance(self, watch=False):
        observation = self.test_environment.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = self.agent.act([observation], True)
            env_action = self.project_actions(action).cpu().numpy()[0]
            observation, reward, done, _ = self.test_environment.step(env_action)
            episode_reward += reward
            if watch:
                self.test_environment.render()
        return episode_reward

    def collect_batch(self, time_steps):
        # self.agent.reset_noise()
        values, log_p_for_actions, batch_entropy, rewards, is_done = [], [], [], [], []
        for step in range(time_steps):
            action, value, log_p_for_action, entropy = self.agent(self.last_observation)
            env_action = self.project_actions(action).cpu().numpy()
            observation, reward, done, _ = self.train_environment.step(env_action)
            self.last_observation = observation
            self.worker_reward += reward
            for i in range(self.num_environments):
                if done[i]:
                    self.episodes_done[i] += 1
                    self.writer.add_scalars(
                        'episode_reward',
                        {'worker_{}'.format(i): self.worker_reward[i]},
                        self.episodes_done[i]
                    )
                    self.worker_reward[i] = 0.0

            values.append(value)
            log_p_for_actions.append(log_p_for_action)
            batch_entropy.append(entropy)
            rewards.append(reward)
            is_done.append(done)
        # append V(s') to the values
        with torch.no_grad():
            _, value, _, _ = self.agent(self.last_observation)
        values.append(value)

        values = torch.stack(values, dim=0)
        log_p_for_actions = torch.stack(log_p_for_actions, dim=0)
        batch_entropy = torch.stack(batch_entropy, dim=0)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        is_done = torch.tensor(is_done, dtype=torch.float32, device=self.device)
        return values, log_p_for_actions, batch_entropy, rewards, is_done

    def compute_gae(self, reward, value):
        step_gae = 0
        gae = []
        for step in reversed(range(len(reward))):
            delta = reward[step] + self.gamma * value[step + 1] - value[step]
            step_gae = delta + self.gamma * self.gae_lambda * step_gae
            gae.append(step_gae)
        return torch.stack(gae[::-1], dim=0)

    def train_on_batch(self, warm_up, batch):
        values, log_p_for_actions, entropy, rewards, is_done = batch

        target_values = (rewards + (1.0 - is_done) * self.gamma * values[1:]).detach()
        if self.use_gae:
            advantage = self.compute_gae(rewards, values)
        else:
            advantage = target_values - values[:-1]
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-4)

        policy_loss = (log_p_for_actions * advantage.detach()).mean()
        value_loss = self.value_loss(values[:-1], target_values)
        entropy = entropy.mean()
        if warm_up:
            loss = value_loss - entropy
        else:
            loss = value_loss - policy_loss - self.entropy_reg * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()

    def train(self, warm_up, num_epochs, num_training_steps, env_steps):
        message = 'training start'
        if warm_up:
            message = message + '. Warm up is on: during 0 epoch policy loss is not optimized'
            add_epoch = 0
        else:
            add_epoch = 1
        print(message)
        print('num_epochs: {}, steps_per_epochs: {}, env_steps: {}'.format(
            num_epochs, num_training_steps, env_steps
        ))
        # test performance before training
        test_reward = sum([self.test_performance() for _ in range(self.num_tests)])
        self.writer.add_scalar('test_reward', test_reward / self.num_tests, 0)
        for epoch in range(num_epochs + add_epoch):
            for train_step in tqdm(range(num_training_steps),
                                   desc='epoch_{}'.format(epoch),
                                   ncols=80):
                batch = self.collect_batch(env_steps)
                policy_loss, value_loss, entropy = self.train_on_batch(epoch + add_epoch == 0, batch)
                # write losses
                step = train_step + epoch * num_training_steps
                self.writer.add_scalar('policy_loss', policy_loss, step)
                self.writer.add_scalar('value_loss', value_loss, step)
                self.writer.add_scalar('entropy', entropy, step)
                self.writer.add_scalar('batch_reward', batch[3].mean(), step)
            # test performance at the epoch end
            test_reward = sum([self.test_performance() for _ in range(self.num_tests)])
            self.writer.add_scalar('test_reward', test_reward / self.num_tests, epoch + 1)
            save(self.logdir + 'epoch_{}.pth'.format(epoch), self.agent)
