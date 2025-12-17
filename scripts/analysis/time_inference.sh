#!/bin/bash
ulimit -n 99999

if [ $# -lt 2 ]; then
    echo "Usage: $0 <cuda_device_id> <model_type>"
    exit 1
fi

# Parse command line arguments
cuda_device_id=$1  # GPU device ID to use for evaluation
model_type=$2      # Model type: 'panda', 'chronos', or 'chronos_sft'

echo "cuda_device_id: $cuda_device_id"
echo "model_type: $model_type"

test_data_dirs=(
    $WORK/data/improved/final_base40/test_zeroshot
    $WORK/data/improved/final_skew40/test_zeroshot
)
test_data_dirs_json=$(printf '%s\n' "${test_data_dirs[@]}" | jq -R . | jq -s -c .)
echo "test_data_dirs: $test_data_dirs_json"

if [ "$model_type" = "panda" ]; then
    run_name=panda-21M
    checkpoint_path=GilpinLab/panda

elif [ "$model_type" = "panda-72M" ]; then
    model_type=panda
    run_name=panda-72M
    checkpoint_path=GilpinLab/panda-72M

elif [ "$model_type" = "chronos_sft" ]; then
    model_type=chronos
    run_name=chronos_t5_mini_ft-0
    checkpoint_path=$WORK/checkpoints/$run_name/checkpoint-final

elif [ "$model_type" = "chronos" ]; then
    run_name=chronos_mini_zeroshot
    checkpoint_path=amazon/chronos-t5-mini

elif [ "$model_type" = "chronos_small" ]; then
    model_type=chronos
    run_name=chronos_small_zeroshot
    checkpoint_path=amazon/chronos-t5-small

elif [ "$model_type" = "chronos_base" ]; then
    model_type=chronos
    run_name=chronos_base_zeroshot
    checkpoint_path=amazon/chronos-t5-base

elif [ "$model_type" = "timesfm" ]; then
    run_name=timesfm
    checkpoint_path=google/timesfm-1.0-200m-pytorch

elif [ "$model_type" = "timemoe" ]; then
    run_name=timemoe
    checkpoint_path=Maple728/TimeMoE-50M

elif [ "$model_type" = "dynamix" ]; then
    run_name=dynamix
    checkpoint_path=null

else
    echo "Unknown model_type: $model_type"
    exit 1
fi

echo "run_name: $run_name"

num_samples_chronos=10
if [ "$model_type" = "chronos" ] && [ "$num_samples_chronos" -gt 1 ]; then
    model_dir="chronos_nondeterministic"
    use_deterministic_chronos=false
else
    model_dir="$model_type"
    use_deterministic_chronos=true
fi

echo "model_dir: $model_dir"

export PYTHONWARNINGS="ignore"

window_start_times=(512)
for idx in "${!window_start_times[@]}"; do
    window_start_time="${window_start_times[$idx]}"
    echo "Index: $idx, window_start_time: $window_start_time"
    python scripts/analysis/time_inference.py \
        eval.mode=predict \
        eval.model_type=$model_type \
        eval.checkpoint_path=$checkpoint_path \
        eval.device=cuda:$cuda_device_id \
        eval.data_paths_lst=$test_data_dirs_json \
        eval.num_subdirs=300 \
        eval.num_samples_per_subdir=4 \
        eval.chronos.deterministic=$use_deterministic_chronos \
        eval.num_samples=$num_samples_chronos \
        eval.metrics_save_dir=$WORK/timer_debug/$model_dir/$run_name \
        eval.metrics_fname=inference_times-start$window_start_time \
        eval.window_start=$window_start_time \
        eval.prediction_length=512 \
        eval.context_length=512 \
        # eval.torch_dtype=bfloat16
done
