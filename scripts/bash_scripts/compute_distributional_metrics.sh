#!/bin/bash
ulimit -n 99999

if [ $# -lt 2 ]; then
    echo "Usage: $0 <cuda_device_id> <model_type>"
    exit 1
fi

cuda_device_id=$1
model_type=$2

echo "cuda_device_id: $cuda_device_id"
echo "model_type: $model_type"

test_data_dirs=(
    $WORK/data/improved/final_base40/test_zeroshot
    $WORK/data/improved/final_base40/test_zeroshot_z5_z10
    $WORK/data/improved/final_skew40/test_zeroshot
    $WORK/data/improved/final_skew40/test_zeroshot_z5_z10
)
test_data_dirs_json=$(printf '%s\n' "${test_data_dirs[@]}" | jq -R . | jq -s -c .)
echo "test_data_dirs: $test_data_dirs_json"

if [ "$model_type" = "panda" ]; then
    run_name=pft_chattn_emb_w_poly-0
elif [ "$model_type" = "chronos" ]; then
    chronos_model_size=mini
    run_name=chronos_${chronos_model_size}_zeroshot
elif [ "$model_type" = "chronos_sft" ]; then
    model_type=chronos
    run_name=chronos_small_ft_equalized-13
else
    echo "Unknown model_type: $model_type"
    exit 1
fi

echo "run_name: $run_name"

# Set zero_shot flag based on whether "zeroshot" appears in run_name
if [[ "$run_name" == *"zeroshot"* ]]; then
    zero_shot_flag="true"
else
    zero_shot_flag="false"
fi


num_samples_chronos=5
if [ "$model_type" = "chronos" ] && [ "$num_samples_chronos" -gt 1 ]; then
    model_dir="chronos_nondeterministic"
else
    model_dir="$model_type"
fi

echo "model_dir: $model_dir"

export PYTHONWARNINGS="ignore"

export PYTHONWARNINGS="ignore"

window_start_times=(512 1024 1536 2048)
for idx in "${!window_start_times[@]}"; do
    window_start_time="${window_start_times[$idx]}"
    echo "Index: $idx, window_start_time: $window_start_time"
    python scripts/analysis/distribution_metrics.py \
        eval.mode=predict \
        eval.model_type=$model_type \
        eval.checkpoint_path=$WORK/checkpoints/$run_name/checkpoint-final \
        eval.device=cuda:$cuda_device_id \
        eval.data_paths_lst=$test_data_dirs_json \
        eval.num_subdirs=null \
        eval.num_samples_per_subdir=null \
        eval.metrics_save_dir=$WORK/eval_results/$model_dir/$run_name/test_zeroshot \
        eval.metrics_fname=distributional_metrics_window-$window_start_time \
        eval.save_forecasts=true \
        eval.save_full_trajectory=true \
        eval.reload_saved_forecasts=false \
        eval.num_processes=100 \
        eval.window_start=$window_start_time \
        eval.prediction_length=512 \
        eval.context_length=512 \
        eval.chronos.zero_shot=$zero_shot_flag \
        eval.metrics_fname_suffix=all \
        eval.dataloader_num_workers=4 \
        eval.batch_size=512
done

