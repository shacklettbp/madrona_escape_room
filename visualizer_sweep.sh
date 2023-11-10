#!/bin/bash

for model in "checkpoints_e0.03_dense3" "checkpoints_e0.05_dense3" "checkpoints_e0.1_dense3" "checkpoints_e0.15_dense3" "checkpoints_e0.05_dense3_lowlr" "checkpoints_e0.1_dense3_lowlr" "checkpoints_e0.15_dense3_lowlr"
do
    echo $model
    for ckpt in 100 200 500 1000 1500 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        echo $ckpt
        MADRONA_MWGPU_KERNEL_CACHE=/tmp/escapecache python scripts/visualizer_log.py --num-worlds 100 --num-steps 200 --fp16 --run-name $model --ckpt-path build/$model/$ckpt.pth --action-dump-path build/dumped_actions --gpu-sim
    done
done
