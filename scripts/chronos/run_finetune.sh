ulimit -n 100000

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
        --nproc-per-node 4 \
        scripts/chronos/train.py \
        wandb.log=false \
        chronos.model_id="amazon/chronos-t5-mini" \
        chronos.model_type=seq2seq \
        chronos.random_init=false \
        chronos.tie_embeddings=true \
        chronos.context_length=512 \
        chronos.prediction_length=64 \
        chronos.num_samples=20 \
        chronos.n_tokens=4096 \
        chronos.n_special_tokens=2 \
        chronos.pad_token_id=0 \
        chronos.eos_token_id=1 \
        chronos.use_eos_token=true \
        chronos.tokenizer_class=MeanScaleUniformBins \
        chronos.tokenizer_kwargs.low_limit=-15.0 \
        chronos.tokenizer_kwargs.high_limit=15.0 \
        chronos.temperature=1.0 \
        chronos.top_k=50 \
        chronos.top_p=1.0 \
        train.max_steps=200_000 \
        train.save_steps=100_000 \
        train.log_steps=1000 \
        shuffle_buffer_length=100_000 \
        train.per_device_train_batch_size=168 \
        train.warmup_ratio=0.1 \
        train.torch_compile=true \
        train.weight_decay=0.0 \
        "$@"
