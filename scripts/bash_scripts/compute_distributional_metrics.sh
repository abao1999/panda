
ulimit -n 99999

if [ $# -lt 3 ]; then
    echo "Usage: $0 <window_start_time> <cuda_device_id> <model_type>"
    exit 1
fi

window_start_time=$1
cuda_device_id=$2
model_type=$3

echo "window_start_time: $window_start_time"
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
    run_name=chronos_t5_mini_ft-0
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


# Limit threads for libraries that auto-parallelize
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/compute_distributional_metrics.py \
    eval.mode=predict \
    eval.model_type=$model_type \
    eval.checkpoint_path=$WORK/checkpoints/$run_name/checkpoint-final \
    eval.device=cuda:$cuda_device_id \
    eval.data_paths_lst=$test_data_dirs_json \
    eval.num_subdirs=null \
    eval.num_samples_per_subdir=null \
    eval.metrics_save_dir=$WORK/eval_results/${model_type}/$run_name/test_zeroshot \
    eval.metrics_fname=distributional_metrics_window-$window_start_time \
    eval.save_forecasts=true \
    eval.save_full_trajectory=true \
    eval.reload_saved_forecasts=true \
    eval.num_processes=100 \
    eval.window_start_time=$window_start_time \
    eval.prediction_length=512 \
    eval.context_length=512 \
    eval.chronos.zero_shot=$zero_shot_flag \


# python scripts/compute_distributional_metrics.py \
#     eval.mode=predict \
#     eval.model_type=$model_type \
#     eval.checkpoint_path=$WORK/checkpoints/$run_name/checkpoint-final \
#     eval.device=cuda:$cuda_device_id \
#     eval.data_paths_lst=$test_data_dirs_json \
#     eval.num_subdirs=null \
#     eval.num_samples_per_subdir=null \
#     eval.metrics_save_dir=$WORK/eval_results/${model_type}_nondeterministic/$run_name/test_zeroshot \
#     eval.metrics_fname=distributional_metrics_window-$window_start_time \
#     eval.save_forecasts=true \
#     eval.save_full_trajectory=true \
#     eval.reload_saved_forecasts=false \
#     eval.num_processes=100 \
#     eval.window_start_time=$window_start_time \
#     eval.prediction_length=512 \
#     eval.context_length=512 \
#     eval.chronos.zero_shot=$zero_shot_flag \
