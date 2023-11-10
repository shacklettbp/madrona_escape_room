#!/bin/bash

# MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/train.py --num-worlds 8192 --num-updates 10000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints_e0.01_dense3/ --run-name e0.01 --entropy-loss-coef 0.01

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/train.py --num-worlds 8192 --num-updates 10000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints_e0.03_dense3/ --run-name e0.03 --entropy-loss-coef 0.03

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/train.py --num-worlds 8192 --num-updates 10000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints_e0.05_dense3/ --run-name e0.05 --entropy-loss-coef 0.05

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/train.py --num-worlds 8192 --num-updates 10000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints_e0.1_dense3/ --run-name e0.1 --entropy-loss-coef 0.1

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/train.py --num-worlds 8192 --num-updates 10000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints_e0.15_dense3/ --run-name e0.15 --entropy-loss-coef 0.15

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/train.py --num-worlds 8192 --num-updates 10000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints_e0.05_dense3_lowlr/ --run-name e0.05 --entropy-loss-coef 0.05 --lr 5e-5

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/train.py --num-worlds 8192 --num-updates 10000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints_e0.1_dense3_lowlr/ --run-name e0.1 --entropy-loss-coef 0.1 --lr 5e-5

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/train.py --num-worlds 8192 --num-updates 10000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints_e0.15_dense3_lowlr/ --run-name e0.15 --entropy-loss-coef 0.15 --lr 5e-5