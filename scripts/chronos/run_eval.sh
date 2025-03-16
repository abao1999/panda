#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints

ulimit -n 99999

split_dir=final_skew40/test_zeroshot

# run_name=chronos_mini_zeroshot
# python scripts/chronos/evaluate.py \
#         eval.checkpoint_path=amazon/chronos-t5-mini \
#         eval.data_path=$WORK/data/copy/${split_dir} \
#         eval.num_systems=null \
#         eval.num_test_instances=1 \
#         eval.window_style=sampled \
#         eval.batch_size=64 \
#         eval.prediction_length=512 \
#         eval.limit_prediction_length=false \
#         eval.metrics_save_dir=$main_dir/eval_results/chronos/${run_name}/${split_dir} \
#         eval.metrics_fname=metrics \
#         eval.overwrite=true \
#         eval.device=cuda:1 \
#         eval.forecast_save_dir=$WORK/data/eval/chronos/${run_name}/${split_dir}/forecasts \
#         eval.labels_save_dir=$WORK/data/eval/chronos/${run_name}/${split_dir}/labels \
#         eval.seed=99 \
#         "$@"

# run_name=chronos_finetune_stand_updated-0
run_name=chronos_bolt_mini-12
python scripts/chronos/evaluate.py \
        eval.checkpoint_path=$checkpoint_dir/${run_name}/checkpoint-final \
        eval.data_path=$WORK/data/copy/${split_dir} \
        eval.num_systems=null \
        eval.num_test_instances=1 \
        eval.window_style=sampled \
        eval.batch_size=64 \
        eval.prediction_length=512 \
        eval.limit_prediction_length=false \
        eval.metrics_save_dir=$main_dir/eval_results/chronos/${run_name}/${split_dir} \
        eval.metrics_fname=metrics \
        eval.overwrite=true \
        eval.device=cuda:2 \
        eval.forecast_save_dir=$WORK/data/eval/chronos/${run_name}/${split_dir}/forecasts \
        eval.labels_save_dir=$WORK/data/eval/chronos/${run_name}/${split_dir}/labels \
        eval.seed=99 \
        "$@"
