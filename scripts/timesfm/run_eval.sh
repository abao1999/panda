#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"

ulimit -n 99999

run_name=timesfm-200m
split_dir=final_skew40/test_zeroshot

python scripts/timesfm/evaluate.py \
        eval.data_path=$WORK/data/improved/${split_dir} \
        eval.num_systems=null \
        eval.num_test_instances=1 \
        eval.window_style=sampled \
        eval.batch_size=64 \
        eval.prediction_length=512 \
        eval.limit_prediction_length=false \
        eval.metrics_save_dir=$WORK/eval_results/timesfm/${run_name}/${split_dir} \
        eval.metrics_fname=metrics \
        eval.overwrite=true \
        eval.device=cuda:1 \
        eval.forecast_save_dir=$WORK/data/eval/timesfm/${run_name}/${split_dir}/forecasts \
        eval.labels_save_dir=$WORK/data/eval/timesfm/${run_name}/${split_dir}/labels \
        eval.seed=99 \
        "$@"
