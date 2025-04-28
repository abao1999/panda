#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints

ulimit -n 99999

split_dir=final_skew40/test_zeroshot

run_name=chronos_zeroshot

# Set zero_shot flag based on whether "zeroshot" appears in run_name
if [[ "$run_name" == *"zeroshot"* ]]; then
    zero_shot_flag="true"
else
    zero_shot_flag="false"
fi

python scripts/chronos/evaluate.py \
        eval.checkpoint_path=$checkpoint_dir/${run_name}/checkpoint-final \
        eval.data_path=$WORK/data/improved/${split_dir} \
        eval.num_systems=null \
        eval.num_test_instances=6 \
        eval.window_style=sampled \
        eval.batch_size=64 \
        eval.prediction_length=512 \
        eval.limit_prediction_length=false \
        eval.metrics_save_dir=$WORK/eval_results/chronos/${run_name}/${split_dir} \
        eval.metrics_fname=metrics \
        eval.overwrite=true \
        eval.device=cuda:3 \
        eval.save_predictions=false \
        eval.save_labels=false \
        eval.forecast_save_dir=$WORK/data/eval/chronos/${run_name}/${split_dir}/forecasts \
        eval.labels_save_dir=$WORK/data/eval/chronos/${run_name}/${split_dir}/labels \
        eval.chronos.zero_shot=$zero_shot_flag \
        eval.seed=99 \
        "${@:2}"  # Pass remaining arguments after run_name
