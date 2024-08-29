#!/bin/bash

run_job_bg() {
    echo "RUNNING: $@"
    python "$@" &
}

# Function to run Python job and wait for it to finish
run_job_bg_wait() {
    echo "RUNNING: $@"
    python "$@" &
    wait $!
}

run_job_seq() {
    echo "RUNNING: $@"
    python "$@" && echo "$@ finished successfully!"
}

# Set variable to main (parent) directory
main_dir=$(dirname "$(dirname "$0")")
checkpoint_dir=$WORK/checkpoints

# # base chronos large model
# python scripts/evaluate.py \
#         eval.model_id=amazon/chronos-t5-large \
#         eval.data_dir=$WORK/data \
#         eval.split=test \
#         eval.output_dir=$main_dir/eval_results \
#         eval.output_fname=test_base_metrics.csv \
#         eval.overwrite=True \
#         eval.device=cuda:7 \

# fine-tuned chronos-dysts large model, intermediate checkpoint
run_job_bg scripts/evaluate.py \
        eval.model_id=$checkpoint_dir/checkpoint-20000 \
        eval.data_dir=$WORK/data \
        eval.split=test \
        eval.output_dir=$main_dir/eval_results \
        eval.output_fname=test_20000_metrics.csv \
        eval.overwrite=True \
        eval.device=cuda:6 \

# fine-tuned chronos-dysts large model, intermediate checkpoint
run_job_bg scripts/evaluate.py \
        eval.model_id=$checkpoint_dir/checkpoint-40000 \
        eval.data_dir=$WORK/data \
        eval.split=test \
        eval.output_dir=$main_dir/eval_results \
        eval.output_fname=test_40000_metrics.csv \
        eval.overwrite=True \
        eval.device=cuda:5 \

# fine-tuned chronos-dysts large model, intermediate checkpoint
run_job_bg scripts/evaluate.py \
        eval.model_id=$checkpoint_dir/checkpoint-60000 \
        eval.data_dir=$WORK/data \
        eval.split=test \
        eval.output_dir=$main_dir/eval_results \
        eval.output_fname=test_60000_metrics.csv \
        eval.overwrite=True \
        eval.device=cuda:4 \

# fine-tuned chronos-dysts large model, intermediate checkpoint
run_job_bg scripts/evaluate.py \
        eval.model_id=$checkpoint_dir/checkpoint-80000 \
        eval.data_dir=$WORK/data \
        eval.split=test \
        eval.output_dir=$main_dir/eval_results \
        eval.output_fname=test_80000_metrics.csv \
        eval.overwrite=True \
        eval.device=cuda:3 \

# fine-tuned chronos-dysts large model, final checkpoint
run_job_bg scripts/evaluate.py \
        eval.model_id=$checkpoint_dir/checkpoint-final \
        eval.data_dir=$WORK/data \
        eval.split=test \
        eval.output_dir=$main_dir/eval_results \
        eval.output_fname=test_100000_metrics.csv \
        eval.overwrite=True \
        eval.device=cuda:2 \