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

elif [ "$model_type" = "chronos" ]; then
    run_name=chronos_mini_zeroshot
    checkpoint_path=amazon/chronos-t5-mini

elif [ "$model_type" = "chronos_sft" ]; then
    model_type=chronos
    run_name=chronos_t5_mini_ft-0
    checkpoint_path=$WORK/checkpoints/$run_name/checkpoint-final

elif [ "$model_type" = "dynamix" ]; then
    run_name=dynamix
    checkpoint_path=null

else
    echo "Unknown model_type: $model_type"
    exit 1
fi

echo "run_name: $run_name"

num_samples_chronos=5
if [ "$model_type" = "chronos" ] && [ "$num_samples_chronos" -gt 1 ]; then
    model_dir="chronos_nondeterministic"
else
    model_dir="$model_type"
fi

echo "model_dir: $model_dir"

export PYTHONWARNINGS="ignore"

compute_metrics_intervals=(1024)
compute_metrics_intervals_json=$(printf '%s\n' "${compute_metrics_intervals[@]}" | jq -R . | jq -s -c .)
echo "compute_metrics_intervals: $compute_metrics_intervals_json"

window_start_times=(1536 2048)
for idx in "${!window_start_times[@]}"; do
    window_start_time="${window_start_times[$idx]}"
    echo "Index: $idx, window_start_time: $window_start_time"
    python scripts/analysis/distribution_metrics_combined.py \
        eval.mode=predict \
        eval.model_type=$model_type \
        eval.checkpoint_path=$checkpoint_path \
        eval.device=cuda:$cuda_device_id \
        eval.data_paths_lst=$test_data_dirs_json \
        eval.num_subdirs=null \
        eval.num_samples_per_subdir=null \
        eval.metrics_save_dir=$WORK/eval_results_distributional/$model_dir/$run_name/test_zeroshot \
        eval.metrics_fname=distributional_metrics_window-$window_start_time \
        eval.save_forecasts=true \
        eval.save_full_trajectory=true \
        eval.compute_distributional_metrics=true \
        eval.distributional_metrics_predlengths=$compute_metrics_intervals_json \
        eval.reload_saved_forecasts=true \
        eval.num_processes=10 \
        eval.window_start=$window_start_time \
        eval.prediction_length=1024 \
        eval.context_length=512 \
        eval.metrics_fname_suffix=all \
        eval.dataloader_num_workers=0 \
        eval.batch_size=512
done

