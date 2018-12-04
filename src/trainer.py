import torch
from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(self,
                 train_environment, num_environments, test_environment, num_tests,
                 agent, distribution, device, optimizer, value_loss,
                 entropy_reg, gamma, gae_lambda, normalize_advantage,
                 use_gae, ppo_eps, num_ppo_epochs, ppo_batch_size,
                 writer):
        # environments
        self.train_environment = train_environment
        self.num_environments = num_environments
        self.test_environment = test_environment
        self.num_tests = num_tests
        self.last_observation = self.train_environment.reset()

        # agent and optimizer
        self.agent = agent
        self.distribution = distribution
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
        self.ppo_eps = ppo_eps
        self.num_ppo_epochs = num_ppo_epochs
        self.ppo_batch_size = ppo_batch_size

        # writer and statistics
        self.writer = writer
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
        print('trainer initialized')
        print('training parameters:')
        print('\t value_loss: {}, gamma: {}, entropy_reg: {}'.format(vl, self.gamma, self.entropy_reg))
        print('\t use_gae: {}, gae_lambda: {}, normalize_adv: {}'.format(
            self.use_gae, self.gae_lambda, self.normalize_advantage
        ))
        print('\t ppo_batch_size: {}, ppo_epochs: {}'.format(self.ppo_batch_size, self.num_ppo_epochs))

    def project_actions(self, actions):
        return self.discrete_actions[actions]

    def test_performance(self, watch=False):
        observation = self.test_environment.reset()
        done = False
        episode_reward = 0.0
        while not done:
            with torch.no_grad():
                action = self.agent.act([observation], greedy=True)
            env_action = self.project_actions(action).cpu().numpy()
            observation, reward, done, _ = self.test_environment.step(env_action)
            episode_reward += reward
            if watch:
                self.test_environment.render()
        return episode_reward

    def collect_batch(self, time_steps):
        self.agent.reset_noise()
        observations, actions, rewards, is_done = [self.last_observation], [], [], []
        for step in range(time_steps):
            with torch.no_grad():
                action = self.agent.act(self.last_observation)
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

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            is_done.append(done)
        observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        actions = torch.stack(actions, dim=0)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        is_done = torch.tensor(is_done, dtype=torch.float32, device=self.device)
        return observations, actions, rewards, is_done

    def compute_gae(self, reward, value, next_value):
        step_gae = 0
        gae = []
        for step in reversed(range(len(reward))):
            delta = reward[step] + self.gamma * next_value[step] - value[step]
            step_gae = delta + self.gamma * self.gae_lambda * step_gae
            gae.append(step_gae)
        return torch.stack(gae[::-1], dim=0)

    def train_on_batch(self, warm_up, batch):
        observations, actions, rewards, is_done = batch
        observations, next_observations = observations[:-1], observations[1:]
        time, batch = rewards.size()
        obs_size = observations.size()[2:]

        # compute log_pi_old, V(s), V(s')
        with torch.no_grad():
            old_log_p, values, _ = self.agent.log_p_for_action(
                observations.view(time * batch, *obs_size),
                actions.view(time * batch)
            )
            _, next_values = self.agent.policy(
                next_observations.view(time * batch, *obs_size)
            )
        # compute advantage with GAE or TD difference
        if self.use_gae:
            advantage = self.compute_gae(rewards,
                                         values.view(time, batch),
                                         (1.0 - is_done) * next_values.view(time, batch))
        else:
            temp = rewards + self.gamma * (1.0 - is_done) * next_values.view(time, batch)
            advantage = temp - values.view(time, batch)
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-4)

        # run PPO epochs
        policy_loss, value_loss, entropy = 0., 0., 0.
        for _ in range(self.num_ppo_epochs):
            # form batch
            ppo_batch = self.form_ppo_batch(batch, time, obs_size,
                                            observations, actions, rewards,
                                            is_done, next_observations,
                                            advantage, old_log_p)
            # train on batch and update statistics
            epoch_policy_loss, epoch_value_loss, epoch_entropy = self.ppo_epoch(warm_up, *ppo_batch)
            policy_loss += epoch_policy_loss
            value_loss += epoch_value_loss
            entropy += epoch_entropy
        d = self.num_ppo_epochs  # 'd' stands for 'denominator'
        return policy_loss / d, value_loss / d, entropy / d

    def form_ppo_batch(self, batch, time, obs_size,
                       observations, actions, rewards, is_done,
                       next_observations, advantage, old_log_p):
        indices = torch.randint(0, batch * time, size=(self.ppo_batch_size,), dtype=torch.long)
        return observations.view(-1, *obs_size)[indices], actions.view(-1)[indices], \
               rewards.view(-1)[indices], is_done.view(-1)[indices], \
               next_observations.view(-1, *obs_size)[indices], \
               advantage.view(-1)[indices], old_log_p.view(-1)[indices]

    def ppo_epoch(self, warm_up,
                  observations, actions, rewards, is_done,
                  next_observations, advantage, old_log_p):
        log_p, value, entropy = self.agent.log_p_for_action(observations, actions)
        ratio = (log_p - old_log_p).exp()
        surrogate1 = ratio * advantage
        surrogate2 = torch.clamp(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps) * advantage
        policy_loss = torch.min(surrogate1, surrogate2).mean()
        with torch.no_grad():
            _, next_value = self.agent.policy(next_observations)
        target_value = rewards + self.gamma * (1.0 - is_done) * next_value
        value_loss = self.value_loss(value, target_value)
        entropy = entropy.mean()

        if warm_up:
            loss = value_loss - self.entropy_reg * entropy
        else:
            # may be we should balance value and policy losses
            loss = value_loss - policy_loss - self.entropy_reg * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()

    def train(self, warm_up, num_epochs, num_training_steps, env_steps):
        message = 'training start'
        if warm_up:
            message = message + '. Warm up is on: during 0 epoch policy loss is not optimized'
            start_epoch = 0
        else:
            start_epoch = 1
        print(message)
        print('num_epochs: {}, steps_per_epochs: {}, env_steps: {}'.format(
            num_epochs, num_training_steps, env_steps
        ))
        # test performance before training
        test_reward = sum([self.test_performance() for _ in range(self.num_tests)])
        self.writer.add_scalar('test_reward', test_reward / self.num_tests, 0)
        for epoch in range(start_epoch, num_epochs + 1):
            for train_step in tqdm(range(num_training_steps),
                                   desc='epoch_{}'.format(epoch),
                                   ncols=80):
                batch = self.collect_batch(env_steps)
                policy_loss, value_loss, entropy = self.train_on_batch(epoch == 0, batch)
                # write losses
                step = train_step + epoch * num_training_steps
                self.writer.add_scalar('policy_loss', policy_loss, step)
                self.writer.add_scalar('value_loss', value_loss, step)
                self.writer.add_scalar('entropy', entropy, step)
                self.writer.add_scalar('batch_reward', batch[2].mean(), step)
            # test performance at the epoch end
            test_reward = sum([self.test_performance() for _ in range(self.num_tests)])
            self.writer.add_scalar('test_reward', test_reward / self.num_tests, epoch + 1)
