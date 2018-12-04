import torch
import argparse

from src import create_env, Net, Agent, Trainer

if __name__ == '__main__':
    def bool_arg(x_str):
        x_str = x_str.lower()
        if x_str not in ['true', 'false']:
            raise BaseException('bool argument should be either \'true\' or \'false\', provided {}'.format(x_str))
        return x_str == 'true'


    optimizers = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        'RMSprop': torch.optim.RMSprop
    }

    print('====================== training script ======================')
    parser = argparse.ArgumentParser(description='Policy gradient runner')
    # environment parameters
    parser.add_argument('--num-envs', type=int, default=16,
                        help='number of parallel environments for training (default: 16)')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='number frames to stack in one observation (default: 5)')
    parser.add_argument('--frame-skip', type=int, default=5,
                        help='number of skipped frames per step (default: 5)')
    # agent parameters
    parser.add_argument('--noisy', type=bool_arg, default='false',
                        help='if true then network use noisy linear layers (default: false)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer name, one from {SGD, Adam, RMSprop} (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    # logdir
    parser.add_argument('--logdir', type=str, default='logs/default/',
                        help='directory to store training logs (default: \'logs/default/\')')
    # trainer params
    parser.add_argument('--num_tests', type=int, default=3,
                        help='number of tests per epoch (default: 3)')
    parser.add_argument('--value_loss', type=str, default='mse',
                        help='type of value loss, one from {mse, huber} (default: mse)')
    parser.add_argument('--entropy', type=float, default=1e-3,
                        help='entropy regularization coefficient (default: 1e-3)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor (default: 0.99)')
    parser.add_argument('--use_gae', type=bool_arg, default='false',
                        help='if true then advantage estimated with GAE (default: false)')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='lambda parameter for GAE (default: 0.95)')
    parser.add_argument('--normalize_adv', type=bool_arg, default='false',
                        help='if true then advantage normalized over batch and time dimensions (default: false)')
    # ppo parameters
    parser.add_argument('--ppo_eps', type=float, default=0.1,
                        help='ppo clipping parameter (default: 0.1)')
    parser.add_argument('--ppo_epochs', type=int, default=5,
                        help='number of ppo optimization epochs (default: 5)')
    parser.add_argument('--ppo_batch', type=int, default=40,
                        help='ppo batch size (default: 40)')
    # training parameters
    parser.add_argument('warm_up', type=bool_arg, default='false',
                        help='if true then critic will be trained one additional epoch before actor (default: false)')
    parser.add_argument('num_epochs', type=int, default=100,
                        help='number of training epochs (default: 100)')
    parser.add_argument('steps_per_epoch', type=int, default=500,
                        help='number of training steps per epoch (default: 500)')
    parser.add_argument('env_steps', type=int, default=10,
                        help='number of environment steps per training step (default: 10)')

    args = parser.parse_args()

    # initialize environment
    vec_env, env = create_env(args.num_envs, args.num_frames, args.frame_skip)

    # initialize agent
    net = Net(args.num_frames, args.noisy)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    agent = Agent(net, device)
    optimizer = optimizers[args.optimizer](net.parameters(), args.lr)

    # initialize trainer
    trainer = Trainer(vec_env, args.num_envs, env, args.num_tests,
                      agent, device, optimizer, args.value_loss,
                      args.entropy, args.gamma, args.gae_lambda,
                      args.normalize_adv, args.use_gae,
                      args.ppo_eps, args.ppo_epochs, args.ppo_batch,
                      args.logdir)
    print('======================= start training ======================')
    warm_up = False
    trainer.train(warm_up, 20, 150, 10)
    print('======================= training done =======================')
    print('models saved in {}'.format(args.logdir))
