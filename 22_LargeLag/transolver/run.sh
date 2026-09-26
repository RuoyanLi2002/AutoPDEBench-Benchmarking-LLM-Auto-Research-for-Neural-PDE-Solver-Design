#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10

python main.py \
    --config configs/config.yaml \
    --to_train