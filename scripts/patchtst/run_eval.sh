main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints

# /stor/work/AMDG_Gilpin_Summer2024/checkpoints/run-179/checkpoint-final

# mlm pretrain eval
python scripts/patchtst/evaluate.py \
    patchtst.patch_length=16 \
    patchtst.patch_stride=16 \
    patchtst.num_hidden_layers=8 \
    patchtst.num_attention_heads=8 \
    patchtst.d_model=512 \
    patchtst.num_parallel_samples=1 \
    patchtst.context_length=512 \
    patchtst.use_channel_embedding=false \
    eval.mode=pretrain \
    eval.data_path=$WORK/data/big_flow_skew_systems \
    eval.checkpoint_path=$WORK/checkpoints/run-281/checkpoint-final \
    eval.num_test_instances=1 \
    eval.batch_size=64 \
    eval.metrics_save_dir=$main_dir/eval_results \
    eval.metrics_fname=eval_mlm_metrics \
    eval.overwrite=true \
    eval.device=cuda:6 \
    eval.forecast_save_dir=$WORK/data/eval/forecasts \
    eval.completions_save_dir=$WORK/data/eval/completions \
    eval.patch_input_save_dir=$WORK/data/eval/patch_input \
    noiser.start=0.0 \
    noiser.enabled=false \
    quantizer.enabled=false \
    # eval.window_style=rolling \
    # eval.window_stride=1 \
    # eval.num_test_instances=1 \


    # eval.num_samples=null \
    # eval.agg_axis=null \


# # prediction finetune eval
# python scripts/patchtst/evaluate.py \
#     patchtst.context_length=512 \
#     patchtst.prediction_length=64 \
#     patchtst.patch_length=16 \
#     patchtst.patch_stride=16 \
#     patchtst.num_hidden_layers=8 \
#     patchtst.num_attention_heads=8 \
#     patchtst.d_model=512 \
#     patchtst.num_parallel_samples=1 \
#     eval.limit_prediction_length=false \
#     eval.prediction_length=3584 \
#     eval.data_path=$WORK/data/backup_test \
#     eval.checkpoint_path=$WORK/checkpoints/run-182/checkpoint-final \
#     eval.window_style=sampled \
#     eval.window_stride=64 \
#     eval.num_test_instances=1 \
#     eval.batch_size=64 \
#     eval.output_dir=$main_dir/eval_results \
#     eval.output_fname=$main_dir/test_finetune_metrics.csv \
#     eval.overwrite=true \
#     eval.device=cuda:6 \
#     eval.forecast_save_dir=$WORK/data/forecasts/test \
#     "$@"

# # old prediction finetune eval
# python scripts/patchtst/evaluate_old.py \
#     patchtst.patch_length=16 \
#     patchtst.patch_stride=16 \
#     patchtst.num_hidden_layers=8 \
#     patchtst.num_attention_heads=8 \
#     patchtst.d_model=512 \
#     patchtst.num_parallel_samples=1 \
#     patchtst.context_length=512 \
#     patchtst.prediction_length=64 \
#     eval.limit_prediction_length=false \
#     eval.prediction_length=3584 \
#     eval.offset=-3584 \
#     eval.data_path=$WORK/data/backup_test \
#     eval.checkpoint_path=$WORK/checkpoints/run-182/checkpoint-final \
#     eval.batch_size=64 \
#     eval.output_dir=$main_dir/eval_results \
#     eval.output_fname=$main_dir/test_finetune_metrics.csv \
#     eval.overwrite=true \
#     eval.device=cuda:6 \
#     eval.forecast_save_dir=$WORK/data/forecasts/test \
#     eval.num_samples=null \
#     eval.agg_axis=null \
#     eval.save_forecasts_to_npy=true \
#     "$@"
