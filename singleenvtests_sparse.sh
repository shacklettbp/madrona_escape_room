#!/bin/bash

#MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/train.py --num-worlds 8192 --num-updates 10000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints_e0.01_sparse/ --run-name e0.01_sparse --entropy-loss-coef 0.01 --num-bptt-chunks 1

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/train.py --num-worlds 8192 --num-updates 10000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints_e0.03_sparse/ --run-name e0.03_sparse --entropy-loss-coef 0.03 --num-bptt-chunks 1

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/train.py --num-worlds 8192 --num-updates 10000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints_e0.1_sparse/ --run-name e0.1_sparse --entropy-loss-coef 0.1 --num-bptt-chunks 1

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/train.py --num-worlds 8192 --num-updates 10000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints_e0.3_sparse/ --run-name e0.3_sparse --entropy-loss-coef 0.3 --num-bptt-chunks 1

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/train.py --num-worlds 8192 --num-updates 10000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints_e0.03_lr1e-5_sparse/ --run-name e0.03_lr1e-5_sparse --entropy-loss-coef 0.03 --lr 1e-5 --num-bptt-chunks 1

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/train.py --num-worlds 8192 --num-updates 10000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints_e0.1_lr1e-5_sparse/ --run-name e0.1_lr1e-5_sparse --entropy-loss-coef 0.1 --lr 1e-5 --num-bptt-chunks 1