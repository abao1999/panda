#!/usr/bin/env bash
set -euo pipefail
ulimit -n 99999

main_dir=$(cd "$(dirname "$0")/../.." && pwd)

test_data_dirs=(
    $WORK/data/improved/final_base40/test_zeroshot
    $WORK/data/improved/final_base40/test_zeroshot_z5_z10
    $WORK/data/improved/final_skew40/test_zeroshot
    $WORK/data/improved/final_skew40/test_zeroshot_z5_z10
)
test_data_dirs_json=$(printf '%s\n' "${test_data_dirs[@]}" | jq -R . | jq -s -c .)

python "$main_dir/scripts/dynamix/evaluate.py" \
    eval.mode=predict \
    eval.data_paths_lst="$test_data_dirs_json" \
    eval.num_subdirs=null \
    eval.num_test_instances=6 \
    eval.window_style=sampled \
    eval.batch_size=32 \
    eval.prediction_length=512 \
    eval.limit_prediction_length=false \
    eval.metrics_save_dir="$WORK/eval_results/dynamix/test_zeroshot" \
    eval.forecast_save_dir="$WORK/eval_results/dynamix/test_zeroshot/forecasts" \
    eval.labels_save_dir="$WORK/eval_results/dynamix/test_zeroshot/labels" \
    eval.metrics_fname=metrics \
    eval.overwrite=true \
    eval.device=cuda:0 \
    eval.save_forecasts=true \
    eval.save_labels=true \
    eval.seed=99 \
    dynamix.model_name=dynamix-3d-alrnn-v1.0 \
    "$@"
