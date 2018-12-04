#!/bin/bash

# num_env=8, noisy, Adam, lr=2.5e-4, use_gae=True, ppo_batch=20
# entropy=1e-3
# false, 20, 150, 10

python3 train.py \
--num-envs 8 --num_frames 5 --frame_skip 5 --wrap_reward false \
--noisy true --optimizer Adam --lr 2.5e-4 \
--logdir logs/trial-0/ \
--num_tests 3 --value_loss mse --entropy 1e-3 \
--gamma 0.99 --use_gae true --gae_lambda 0.95 --normalize_adv false \
--ppo_eps 0.1 --ppo_epochs 5 --ppo_batch 20 \
false 20 150 10