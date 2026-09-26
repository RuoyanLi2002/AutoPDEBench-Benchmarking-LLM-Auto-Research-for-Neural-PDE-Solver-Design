#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10

python main.py \
    --config  \
    --eval_split "test"