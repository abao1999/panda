#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints

ulimit -n 99999

run_names=(
    pft_linattnpolyemb_from_scratch-0
)

split_dir=final_skew40/test_zeroshot

use_sliding_context=true
model_dirname=patchtst
if [ "$use_sliding_context" = true ]; then
    model_dirname=patchtst_sliding
    echo "Using sliding context"
fi

for run_name in ${run_names[@]}; do
    echo "Evaluating $run_name"
    python scripts/patchtst/evaluate.py \
        eval.mode=predict \
        eval.sliding_context=$use_sliding_context \
        eval.checkpoint_path=$checkpoint_dir/$run_name/checkpoint-final \
        eval.data_path=$WORK/data/improved/$split_dir \
        eval.num_systems=null \
        eval.num_test_instances=5 \
        eval.window_style=sampled \
        eval.batch_size=64 \
        eval.context_length=512 \
        eval.prediction_length=512 \
        eval.limit_prediction_length=false \
        eval.metrics_save_dir=$WORK/eval_results/$model_dirname/$run_name/$split_dir \
        eval.metrics_fname=metrics \
        eval.overwrite=true \
        eval.device=cuda:2 \
        eval.forecast_save_dir=$WORK/data/eval/$model_dirname/$run_name/$split_dir/forecasts \
        eval.labels_save_dir=$WORK/data/eval/$model_dirname/$run_name/$split_dir/labels \
        fixed_dim=3 \
        eval.seed=99 \
        "$@"
done