#!/bin/bash

python3 train.py \
--num-envs 8 --num_frames 4 --frame_skip 2 --wrap_reward false \
--noisy false --optimizer SGD --lr 2.5e-4 \
--logdir logs/a2c/conv_net/ \
--num_tests 3 --value_loss huber --entropy 1e-1 \
--gamma 0.1 --use_gae false --gae_lambda 0.95 --normalize_adv true \
false 20 500 2
