#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints
# # chronos zero-shot
# eval.checkpoint_path=amazon/chronos-t5-mini \

run_num=357
# fine-tuned dystformer large model, final checkpoint
python scripts/chronos/evaluate.py \
        eval.checkpoint_path=$checkpoint_dir/run-${run_num}/checkpoint-final \
        eval.data_path=$WORK/data/final_skew15/test_base \
        eval.num_systems=10 \
        eval.num_test_instances=1 \
        eval.window_style=sampled \
        eval.batch_size=64 \
        eval.metrics_save_dir=$main_dir/eval_results/chronos \
        eval.metrics_fname=zeroshot_forecast_${run_num}_metrics \
        eval.overwrite=true \
        eval.device=cuda:1 \
        eval.forecast_save_dir=$WORK/data/eval/chronos/forecasts/run-${run_num} \
        eval.labels_save_dir=$WORK/data/eval/chronos/labels/run-${run_num} \
        eval.seed=99 \
        "$@"