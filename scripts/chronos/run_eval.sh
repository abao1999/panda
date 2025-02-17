#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints
# # chronos zero-shot
# eval.checkpoint_path=amazon/chronos-t5-mini \

run_num=387
python scripts/chronos/evaluate.py \
        eval.checkpoint_path=$checkpoint_dir/run-${run_num}/checkpoint-200000 \
        eval.data_path=$WORK/data/final_skew40/test_zeroshot \
        eval.num_systems=1 \
        eval.num_test_instances=1 \
        eval.window_style=sampled \
        eval.batch_size=64 \
        eval.prediction_length=512 \
        eval.limit_prediction_length=false \
        eval.metrics_save_dir=$main_dir/eval_results/chronos/${run_num}_metrics/zeroshot \
        eval.metrics_fname=null \
        eval.overwrite=true \
        eval.device=cuda:0 \
        eval.forecast_save_dir=$WORK/data/eval/chronos/run-${run_num}/forecasts \
        eval.labels_save_dir=$WORK/data/eval/chronos/run-${run_num}/labels \
        eval.seed=99 \
        "$@"
