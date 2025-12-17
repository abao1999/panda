#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"

ulimit -n 99999

run_name=timesfm-200m
test_data_dirs=(
    $WORK/data/improved/final_base40/test_zeroshot
    $WORK/data/improved/final_skew40/test_zeroshot
)
test_data_dirs_json=$(printf '%s\n' "${test_data_dirs[@]}" | jq -R . | jq -s -c .)
echo "test_data_dirs: $test_data_dirs_json"

python scripts/timesfm/evaluate.py \
    eval.data_paths_lst=$test_data_dirs_json \
    eval.num_subdirs=null \
    eval.num_test_instances=6 \
    eval.window_style=sampled \
    eval.batch_size=64 \
    eval.prediction_length=512 \
    eval.limit_prediction_length=false \
    eval.metrics_save_dir=$WORK/eval_results/timesfm/${run_name}/test_zeroshot \
    eval.metrics_fname=metrics \
    eval.overwrite=true \
    eval.device=cuda:3 \
    eval.save_forecasts=false \
    eval.save_labels=false \
    eval.seed=99 \
    "$@"
