#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints

ulimit -n 99999

test_data_dirs=(
    $WORK/data/improved/final_base40/test_zeroshot
    $WORK/data/improved/final_base40/test_zeroshot_z5_z10
    $WORK/data/improved/final_skew40/test_zeroshot
    $WORK/data/improved/final_skew40/test_zeroshot_z5_z10
)
test_data_dirs_json=$(printf '%s\n' "${test_data_dirs[@]}" | jq -R . | jq -s -c .)
echo "test_data_dirs: $test_data_dirs_json"

run_name=chronos_mini_zeroshot

# Set zero_shot flag based on whether "zeroshot" appears in run_name
if [[ "$run_name" == *"zeroshot"* ]]; then
    zero_shot_flag="true"
else
    zero_shot_flag="false"
fi

python scripts/chronos/evaluate.py \
    eval.checkpoint_path=$checkpoint_dir/${run_name}/checkpoint-final \
    eval.data_paths_lst=$test_data_dirs_json \
    eval.num_subdirs=null \
    eval.num_test_instances=6 \
    eval.window_style=sampled \
    eval.batch_size=64 \
    eval.prediction_length=512 \
    eval.limit_prediction_length=false \
    eval.metrics_save_dir=$WORK/eval_results/chronos/${run_name}/test_zeroshot \
    eval.metrics_fname=metrics \
    eval.overwrite=true \
    eval.device=cuda:3 \
    eval.save_predictions=false \
    eval.save_labels=false \
    eval.chronos.zero_shot=$zero_shot_flag \
    eval.seed=99 \
    "${@:2}"  # Pass remaining arguments after run_name
