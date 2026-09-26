#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

python main.py \
    --config configs/config.yaml \
    --eval_split test