#!/bin/bash

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/visualizer_log.py --num-worlds 100 --num-steps 150 --fp16 --ckpt-path build/checkpoints_bptt1/100.pth --action-dump-path build/dumped_actions --gpu-sim

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/visualizer_log.py --num-worlds 100 --num-steps 150 --fp16 --ckpt-path build/checkpoints_bptt1/500.pth --action-dump-path build/dumped_actions --gpu-sim

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/visualizer_log.py --num-worlds 100 --num-steps 150 --fp16 --ckpt-path build/checkpoints_bptt1/1000.pth --action-dump-path build/dumped_actions --gpu-sim

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/visualizer_log.py --num-worlds 100 --num-steps 150 --fp16 --ckpt-path build/checkpoints_bptt1/2000.pth --action-dump-path build/dumped_actions --gpu-sim

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/visualizer_log.py --num-worlds 100 --num-steps 150 --fp16 --ckpt-path build/checkpoints_bptt1/5000.pth --action-dump-path build/dumped_actions --gpu-sim

MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/visualizer_log.py --num-worlds 100 --num-steps 150 --fp16 --ckpt-path build/checkpoints_bptt1/10000.pth --action-dump-path build/dumped_actions --gpu-sim