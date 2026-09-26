#!/bin/bash

python run_agent.py \
        --num_iterations 5 \
        --run_dir agent_runs/run_0 \
        --model_name gpt-5.5 \
        --gpu_id 1 \
        --gpu_memory_gb 48 \
        --max_debug_attempts 5