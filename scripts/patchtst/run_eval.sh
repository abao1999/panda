main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints

# /stor/work/AMDG_Gilpin_Summer2024/checkpoints/run-179/checkpoint-final

# mlm pretrain eval
python scripts/patchtst/evaluate.py \
    patchtst.context_length=512 \
    eval.mode=pretrain \
    eval.data_path=$WORK/data/test_zero_shot \
    eval.checkpoint_path=$WORK/checkpoints/run-303/checkpoint-final \
    eval.num_test_instances=1 \
    eval.batch_size=64 \
    eval.metrics_save_dir=$main_dir/eval_results \
    eval.metrics_fname=eval_mlm_metrics \
    eval.overwrite=true \
    eval.device=cuda:1 \
    eval.forecast_save_dir=$WORK/data/eval/forecasts \
    eval.completions_save_dir=$WORK/data/eval/completions \
    eval.patch_input_save_dir=$WORK/data/eval/patch_input \
    noiser.start=0.0 \
    noiser.enabled=false \
    quantizer.enabled=false \


# # prediction finetune eval
# python scripts/patchtst/evaluate.py \
#     patchtst.patch_length=16 \
#     patchtst.patch_stride=4 \
#     patchtst.num_hidden_layers=8 \
#     patchtst.num_attention_heads=8 \
#     patchtst.context_length=512 \
#     patchtst.prediction_length=128 \
#     patchtst.d_model=512 \
#     patchtst.num_parallel_samples=1 \
#     eval.mode=predict \
#     eval.limit_prediction_length=false \
#     eval.prediction_length=128 \
#     eval.data_path=$WORK/data/test_zero_shot \
#     eval.checkpoint_path=$WORK/checkpoints/run-301/checkpoint-final \
#     eval.num_test_instances=1 \
#     eval.window_style=sampled \
#     eval.batch_size=64 \
#     eval.metrics_save_dir=$main_dir/eval_results \
#     eval.metrics_fname=eval_finetune_metrics \
#     eval.overwrite=true \
#     eval.device=cuda:6 \
#     eval.forecast_save_dir=$WORK/data/eval/forecasts \
#     eval.labels_save_dir=$WORK/data/eval/labels \
#     noiser.start=0.0 \
#     noiser.enabled=false \
#     quantizer.enabled=false \
#     "$@"

#     # eval.window_style=rolling \
#     # eval.window_stride=640 \