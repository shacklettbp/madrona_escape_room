#!/bin/bash

for model in "checkpoints_e0.1_dense3" #"checkpoints_e0.3_lr1e-5_sparse" #"checkpoints_e0.1_sparse" "checkpoints_e0.3_sparse"
do
    echo $model
    for ckpt in 100 200 500 1000 1500 2000 2500 2700
    do
        echo $ckpt
        MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/visualizer_log.py --num-worlds 100 --num-steps 200 --fp16 --run-name $model --ckpt-path build/$model/$ckpt.pth --action-dump-path build/dumped_actions --gpu-sim
    done
done
