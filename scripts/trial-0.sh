#!/bin/bash

python3 train.py \
--num-envs 8 --num_frames 5 --frame_skip 5 --wrap_reward false \
--noisy false --optimizer RMSprop --lr 1e-3 \
--logdir logs/a2c/trial-0/ \
--num_tests 3 --value_loss mse --entropy 1e-1 \
--gamma 0.99 --use_gae false --gae_lambda 0.95 --normalize_adv false \
false 20 150 10
