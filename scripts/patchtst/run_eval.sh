python scripts/patchtst/evaluate.py \
    patchtst.context_length=512 \
    patchtst.prediction_length=64 \
    patchtst.patch_length=16 \
    patchtst.patch_stride=16 \
    patchtst.num_hidden_layers=8 \
    patchtst.num_attention_heads=8 \
    patchtst.d_model=512 \
    eval.data_path=$WORK/data/nonstandardized_test \
    eval.checkpoint_path=$WORK/checkpoints/run-149/checkpoint-final \
    eval.window_style=rolling \
    eval.window_stride=64 \
    eval.num_test_instances=1 \
    eval.batch_size=64 \
    "$@"
