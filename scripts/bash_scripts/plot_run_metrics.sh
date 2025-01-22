#!/bin/bash
run_id=6t8rfen2

python scripts/get_run_metrics.py \
    wandb.entity=gilpinlab \
    wandb.project_name=dystformer \
    run_metrics.wandb_run_id=$run_id \
    run_metrics.plot_dir=figs \
    run_metrics.save_dir=/stor/work/AMDG_Gilpin_Summer2024/data/eval/run_metrics \
    run_metrics.save_fname=${run_id}_metrics.json \
    "$@"

