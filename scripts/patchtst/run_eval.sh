main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints

# /stor/work/AMDG_Gilpin_Summer2024/checkpoints/run-179/checkpoint-final

# # mlm pretrain eval
# python scripts/patchtst/evaluate.py \
#     patchtst.context_length=512 \
#     patchtst.prediction_length=64 \
#     patchtst.patch_length=16 \
#     patchtst.patch_stride=16 \
#     eval.data_path=$WORK/data/nonstandardized_test \
#     eval.checkpoint_path=$WORK/checkpoints/run-179/checkpoint-final \
#     eval.window_style=rolling \
#     eval.window_stride=64 \
#     eval.num_test_instances=1 \
#     eval.batch_size=64 \
#     eval.output_dir=$main_dir/eval_results \
#     eval.output_fname=test_mlm_metrics.csv \
#     eval.overwrite=true \
#     eval.device=cuda:6 \
#     "$@"

# prediction finetune eval
python scripts/patchtst/evaluate.py \
    patchtst.context_length=512 \
    patchtst.prediction_length=64 \
    patchtst.patch_length=16 \
    patchtst.patch_stride=16 \
    patchtst.num_hidden_layers=8 \
    patchtst.num_attention_heads=8 \
    patchtst.d_model=512 \
    patchtst.num_parallel_samples=1 \
    eval.data_path=$WORK/data/test \
    eval.checkpoint_path=$WORK/checkpoints/run-182/checkpoint-final \
    eval.window_style=rolling \
    eval.window_stride=64 \
    eval.num_test_instances=1 \
    eval.batch_size=64 \
    eval.output_dir=$main_dir/eval_results \
    eval.output_fname=$main_dir/test_finetune_metrics.csv \
    eval.overwrite=true \
    eval.device=cuda:6 \
    eval.forecast_save_dir=$WORK/data/forecasts/test \
    "$@"
