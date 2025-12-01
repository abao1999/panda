#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints

ulimit -n 99999

test_data_dirs=(
    $WORK/data/improved/final_base40/test_zeroshot
    $WORK/data/improved/final_skew40/test_zeroshot
)
test_data_dirs_json=$(printf '%s\n' "${test_data_dirs[@]}" | jq -R . | jq -s -c .)
echo "test_data_dirs: $test_data_dirs_json"

chronos_model_size=mini
#run_name=chronos_${chronos_model_size}_zeroshot
run_name=chronos_small_ft_equalized-13
# run_name=chronos_t5_mini_ft-0 # newest chronos sft 300k iterations
# run_name=chronos_small_ft-4

# Set zero_shot flag based on whether "zeroshot" appears in run_name
if [[ "$run_name" == *"zeroshot"* ]]; then
    zero_shot_flag="true"
else
    zero_shot_flag="false"
fi

use_deterministic=false
model_dirname="chronos"
if [ "$use_deterministic" = false ]; then
    model_dirname="chronos_nondeterministic"
fi
echo "model_dirname: $model_dirname"

python scripts/chronos/evaluate.py \
    chronos.model_id=amazon/chronos-t5-${chronos_model_size} \
    chronos.context_length=512 \
    eval.checkpoint_path=$checkpoint_dir/${run_name}/checkpoint-final \
    eval.data_paths_lst=$test_data_dirs_json \
    eval.num_subdirs=null \
    eval.num_test_instances=6 \
    eval.num_samples=10 \
    eval.parallel_sample_reduction=median \
    eval.window_style=sampled \
    eval.batch_size=32 \
    eval.chronos.deterministic=$use_deterministic \
    eval.prediction_length=512 \
    eval.limit_prediction_length=false \
    eval.metrics_save_dir=$WORK/eval_results/${model_dirname}/${run_name}/test_zeroshot \
    eval.metrics_fname=metrics \
    eval.overwrite=true \
    eval.device=cuda:1 \
    eval.save_forecasts=true \
    eval.save_labels=true \
    eval.chronos.zero_shot=$zero_shot_flag \
    eval.dataloader_num_workers=4 \
    eval.seed=99 \
    "$@"
