#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"

name=timesfm-200m
python scripts/timesfm/evaluate.py \
        eval.data_path=$WORK/data/final_skew40/test_base \
        eval.num_systems=325 \
        eval.num_test_instances=3 \
        eval.window_style=sampled \
        eval.batch_size=64 \
        eval.prediction_length=512 \
        eval.limit_prediction_length=false \
        eval.metrics_save_dir=$main_dir/eval_results/timemoe/${name}_metrics/zeroshot \
        eval.metrics_fname=null \
        eval.overwrite=true \
        eval.device=cuda:0 \
        eval.forecast_save_dir=$WORK/data/eval/timemoe/${name}/forecasts \
        eval.labels_save_dir=$WORK/data/eval/timemoe/${name}/labels \
        eval.seed=99 \
        "$@"
