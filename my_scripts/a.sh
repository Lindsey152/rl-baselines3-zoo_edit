#!/bin/bash
echo "Running Test Script"


# python train.py --algo ppo --seed 1 --n-timesteps 10


# python train.py --algo ppo --seed 1 --env CartPole-v1 --n-timesteps 5000 --yaml hyperparams/ppo.yml
# python enjoy.py --algo a2c --env CartPole-v1 --folder logs/


python train.py --algo ppo --env MountainCar-v0 -n 50000 -optimize --n-trials 1000 --n-jobs 2 --sampler tpe --pruner median

# Example of continuing training:
# python train.py --algo ppo --seed 1 --env Pendulum-v1
# python train.py --algo ppo --seed 1 --env Pendulum-v1 --trained-agent logs/ppo/Pendulum-v1_1/Pendulum-v1.zip


# Example of requesting input:
# read -p "Test Input: " testing
# echo $testing