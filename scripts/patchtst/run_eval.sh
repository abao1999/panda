#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints

ulimit -n 99999

# run_name=386
# # mlm pretrain eval
# python scripts/patchtst/evaluate.py \
#     eval.mode=pretrain \
#     eval.checkpoint_path=$checkpoint_dir/run-${run_name}/checkpoint-final \
#     eval.data_path=$WORK/data/final_skew15/test_zeroshot \
#     eval.num_systems=288 \
#     eval.num_test_instances=1 \
#     eval.batch_size=64 \
#     eval.metrics_save_dir=$main_dir/eval_results \
#     eval.metrics_fname=zeroshot_mlm_${run_name}_metrics \
#     eval.overwrite=true \
#     eval.device=cuda:1 \
#     eval.forecast_save_dir=$WORK/data/eval/forecasts/run-${run_name} \
#     eval.completions_save_dir=$WORK/data/eval/completions/run-${run_name} \
#     eval.patch_input_save_dir=$WORK/data/eval/patch_input/run-${run_name} \
#     eval.timestep_masks_save_dir=$WORK/data/eval/timestep_masks/run-${run_name} \
#     use_quadratic_embedding=false \
#     fixed_dim=3 \
#     eval.seed=42 \
#     "$@"

# run_names=(pft_vanilla_pretrained_correct-0 pft_chattn_noembed_pretrained_correct-0 pft_rff_univariate_pretrained-0)
# run_names=(pft_equal_param_deeper_univariate_from_scratch_noemb-0 pft_emb_equal_param_univariate_from_scratch-0)
run_names=(pft_noemb_equal_param_univariate_from_scratch-0)
split_dir=final_skew40/test_zeroshot

for run_name in ${run_names[@]}; do
    echo "Evaluating $run_name"
    python scripts/patchtst/evaluate.py \
        eval.mode=predict \
        eval.checkpoint_path=$checkpoint_dir/$run_name/checkpoint-final \
        eval.data_path=$WORK/data/copy/$split_dir \
        eval.num_systems=null \
        eval.num_test_instances=1 \
        eval.window_style=sampled \
        eval.batch_size=64 \
        eval.prediction_length=512 \
        eval.limit_prediction_length=false \
        eval.metrics_save_dir=$main_dir/eval_results/patchtst/$run_name/$split_dir \
        eval.metrics_fname=metrics \
        eval.overwrite=true \
        eval.device=cuda:3 \
        eval.forecast_save_dir=$WORK/data/eval/patchtst/$run_name/$split_dir/forecasts \
        eval.labels_save_dir=$WORK/data/eval/patchtst/$run_name/$split_dir/labels \
        fixed_dim=3 \
        eval.seed=99 \
        "$@"
done
