#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

python main.py \
    --config configs/ntc_fluid.yaml \
    --eval_split test