#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)

ulimit -n 99999
baselines=(mean) # fourier
# baselines=(fourier_arima)

test_data_dirs=(
    $WORK/data/improved/final_base40/test_zeroshot
    $WORK/data/improved/final_base40/test_zeroshot_z5_z10
    $WORK/data/improved/final_skew40/test_zeroshot
    $WORK/data/improved/final_skew40/test_zeroshot_z5_z10
)
test_data_dirs_json=$(printf '%s\n' "${test_data_dirs[@]}" | jq -R . | jq -s -c .)
echo "test_data_dirs: $test_data_dirs_json"

for baseline in ${baselines[@]}; do
    echo "Evaluating $baseline"
    python scripts/baselines/evaluate.py \
        eval.mode=predict \
        eval.data_paths_lst=$test_data_dirs_json \
        eval.num_subdirs=null \
        eval.num_test_instances=6 \
        eval.window_style=sampled \
        eval.batch_size=64 \
        eval.prediction_length=512 \
        eval.limit_prediction_length=false \
        eval.metrics_save_dir=$WORK/eval_results/baselines/$baseline/test_zeroshot \
        eval.metrics_fname=metrics \
        eval.overwrite=true \
        eval.device=cuda:1 \
        eval.save_forecasts=false \
        eval.save_labels=false \
        eval.seed=99 \
        eval.baselines.baseline_model=$baseline \
        "$@"
done
