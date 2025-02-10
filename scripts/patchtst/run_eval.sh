#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints

# run_num=386
# # mlm pretrain eval
# python scripts/patchtst/evaluate.py \
#     eval.mode=pretrain \
#     eval.checkpoint_path=$checkpoint_dir/run-${run_num}/checkpoint-final \
#     eval.data_path=$WORK/data/final_skew15/test_base \
#     eval.num_systems=288 \
#     eval.num_test_instances=1 \
#     eval.batch_size=64 \
#     eval.metrics_save_dir=$main_dir/eval_results \
#     eval.metrics_fname=zeroshot_mlm_${run_num}_metrics \
#     eval.overwrite=true \
#     eval.device=cuda:1 \
#     eval.forecast_save_dir=$WORK/data/eval/forecasts/run-${run_num} \
#     eval.completions_save_dir=$WORK/data/eval/completions/run-${run_num} \
#     eval.patch_input_save_dir=$WORK/data/eval/patch_input/run-${run_num} \
#     eval.timestep_masks_save_dir=$WORK/data/eval/timestep_masks/run-${run_num} \
#     use_quadratic_embedding=false \
#     fixed_dim=3 \
#     eval.seed=42 \
#     "$@"

run_nums=(391 393 380 400)
# forecast eval
for run_num in "${run_nums[@]}"; do
    echo "Evaluating run ${run_num}"
    python scripts/patchtst/evaluate.py \
        eval.mode=predict \
        eval.checkpoint_path=$checkpoint_dir/run-${run_num}/checkpoint-final \
        eval.data_path=$WORK/data/final_skew40/test_base \
        eval.num_systems=325 \
        eval.num_test_instances=1 \
        eval.window_style=sampled \
        eval.batch_size=64 \
        eval.prediction_length=512 \
        eval.limit_prediction_length=false \
        eval.metrics_save_dir=$main_dir/eval_results/patchtst/${run_num}_metrics/zeroshot \
        eval.metrics_fname=metrics \
        eval.overwrite=true \
        eval.device=cuda:1 \
        eval.forecast_save_dir=$WORK/data/eval/patchtst/run-${run_num}/forecasts \
        eval.labels_save_dir=$WORK/data/eval/patchtst/run-${run_num}/labels \
        use_quadratic_embedding=false \
        fixed_dim=3 \
        eval.seed=99 \
        "$@"
done