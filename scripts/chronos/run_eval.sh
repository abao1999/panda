#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints

ulimit -n 99999

run_num=chronos_finetune_stand_updated-0
split_dir=final_base40/test_zeroshot

python scripts/chronos/evaluate.py \
        eval.checkpoint_path=$checkpoint_dir/${run_num}/checkpoint-final \
        eval.data_path=$WORK/data/copy/${split_dir} \
        eval.num_systems=null \
        eval.num_test_instances=1 \
        eval.window_style=sampled \
        eval.batch_size=64 \
        eval.prediction_length=512 \
        eval.limit_prediction_length=false \
        eval.metrics_save_dir=$main_dir/eval_results/chronos/${run_num}/${split_dir} \
        eval.metrics_fname=metrics \
        eval.overwrite=true \
        eval.device=cuda:0 \
        eval.forecast_save_dir=$WORK/data/eval/chronos/${run_num}/${split_dir}/forecasts \
        eval.labels_save_dir=$WORK/data/eval/chronos/${run_num}/${split_dir}/labels \
        eval.seed=99 \
        "$@"
