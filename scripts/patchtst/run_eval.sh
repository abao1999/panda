main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints

# run_num=367
# # mlm pretrain eval
# python scripts/patchtst/evaluate.py \
#     eval.mode=pretrain \
#     eval.checkpoint_path=$WORK/checkpoints/run-${run_num}/checkpoint-final \
#     eval.data_path=$WORK/data/final_skew15_stationary/test_base \
#     eval.num_systems=288 \
#     eval.num_test_instances=1 \
#     eval.batch_size=64 \
#     eval.metrics_save_dir=$main_dir/eval_results \
#     eval.metrics_fname=zeroshot_mlm_${run_num}_metrics \
#     eval.overwrite=true \
#     eval.device=cuda:1 \
#     eval.forecast_save_dir=$WORK/data/eval/forecasts \
#     eval.completions_save_dir=$WORK/data/eval/completions \
#     eval.patch_input_save_dir=$WORK/data/eval/patch_input \
#     eval.timestep_masks_save_dir=$WORK/data/eval/timestep_masks \
#     use_quadratic_embedding=false \
#     fixed_dim=3 \
#     eval.seed=42 \
#     "$@"

run_num=370 #363
# forecast eval
python scripts/patchtst/evaluate.py \
    eval.mode=predict \
    eval.checkpoint_path=$WORK/checkpoints/run-${run_num}/checkpoint-final \
    eval.data_path=$WORK/data/final_skew15_stationary/test_base \
    eval.num_systems=288 \
    eval.num_test_instances=1 \
    eval.window_style=sampled \
    eval.batch_size=64 \
    eval.metrics_save_dir=$main_dir/eval_results \
    eval.metrics_fname=zeroshot_forecast_${run_num}_metrics \
    eval.overwrite=true \
    eval.device=cuda:1 \
    eval.forecast_save_dir=$WORK/data/eval/forecasts \
    eval.labels_save_dir=$WORK/data/eval/labels \
    use_quadratic_embedding=false \
    fixed_dim=3 \
    eval.seed=99 \
    "$@"