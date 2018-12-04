#!/bin/bash

python3 train.py \
--num-envs 16 --num_frames 5 --frame-skip 5 --wrap-reward false \
--noisy false --optimizer Adam --lr 1e-3 \
--logdir logs/trial-1/ \
--num_tests 3 --value_loss mse --entropy 1e-3 \
--gamma 0.99 --use_gae false --gae_lambda 0.95 --normalize_adv false \
--ppo_eps 0.1 --ppo_epochs 5 --ppo_batch 40 \
warm_up false num_epochs 100 steps_per_epoch 500 env_steps 10
