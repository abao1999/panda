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

if [ "$model_type" = "panda_mlm" ]; then
    run_name=panda_mlm-21M
    checkpoint_path=GilpinLab/panda_mlm

elif [ "$model_type" = "panda_mlm-66M" ]; then
    model_type=panda
    run_name=panda_mlm-66M
    checkpoint_path=GilpinLab/panda_mlm-66M

elif [ "$model_type" = "polyinterp" ]; then
    run_name=polyinterp
    checkpoint_path=null

else
    echo "Unknown model_type: $model_type"
    exit 1
fi


model_dir="$model_type"
echo "model_dir: $model_dir"
echo "run_name: $run_name"

export PYTHONWARNINGS="ignore"

python scripts/analysis/compute_gpdims.py \
    eval.checkpoint_path=$checkpoint_path \
    eval.device=cuda:$cuda_device_id \
    eval.data_paths_lst=$test_data_dirs_json \
    eval.num_subdirs=null \
    eval.num_samples_per_subdir=null \
    eval.metrics_save_dir=$WORK/eval_results_mlm/$model_dir/$run_name/test_zeroshot \
    eval.metrics_fname=gpdims \
    eval.save_completions=true \
    eval.num_processes=10 \
    eval.completions.start_time=0 \
    eval.completions.end_time=null \
    eval.seed=99