#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)

ulimit -n 99999
# baselines=(mean fourier fourier_arima)
baselines=(fourier_arima)
split_dir=final_skew40/test_zeroshot

for baseline in ${baselines[@]}; do
    echo "Evaluating $baseline"
    python scripts/baselines/evaluate.py \
        eval.mode=predict \
        eval.data_path=$WORK/data/improved/$split_dir \
        eval.num_systems=null \
        eval.num_test_instances=1 \
        eval.window_style=sampled \
        eval.batch_size=64 \
        eval.prediction_length=512 \
        eval.limit_prediction_length=false \
        eval.metrics_save_dir=$main_dir/eval_results/baselines/$baseline/$split_dir \
        eval.metrics_fname=metrics \
        eval.overwrite=true \
        eval.device=cuda:2 \
        eval.forecast_save_dir=$WORK/data/eval/baselines/$baseline/$split_dir/forecasts \
        eval.labels_save_dir=$WORK/data/eval/baselines/$baseline/$split_dir/labels \
        eval.seed=99 \
        eval.baselines.baseline_model=$baseline \
        "$@"
done
