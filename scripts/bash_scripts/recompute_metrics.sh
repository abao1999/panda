#! /bin/bash

run_name=pft_fullyfeat_from_scratch-0
split=final_skew40/test_zeroshot

eval_data_dir=$WORK/data/eval/patchtst/${run_name}/${split}
forecast_split="forecasts"
labels_split="labels"

echo "eval_data_dir: $eval_data_dir"
echo "forecast_split: $forecast_split"
echo "labels_split: $labels_split"

metrics_save_dir=$WORK/recomputed_eval_results/patchtst/${run_name}/${split}

python scripts/compute_metrics_from_forecasts.py \
    recompute_metrics.eval_data_dir=$eval_data_dir \
    recompute_metrics.forecast_split=$forecast_split \
    recompute_metrics.labels_split=$labels_split \
    recompute_metrics.num_samples=null \
    recompute_metrics.save_dir=$metrics_save_dir \
