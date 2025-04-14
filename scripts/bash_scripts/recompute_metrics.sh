#! /bin/bash

run_name=pft_fullyfeat_from_scratch-0
split=final_skew40/test_zeroshot

model_dirname=patchtst
use_sliding_context=false

if [ "$use_sliding_context" = true ]; then
    model_dirname=patchtst_sliding
    echo "Using sliding context"
fi

eval_data_dir=$WORK/data/eval/${model_dirname}/${run_name}/${split}

echo "eval_data_dir: $eval_data_dir"

forecast_split="forecasts"
labels_split="labels"

echo "eval_data_dir: $eval_data_dir"
echo "forecast_split: $forecast_split"
echo "labels_split: $labels_split"

metrics_save_dir=$WORK/recomputed_eval_results/patchtst/${run_name}/${split}
echo "Saving metrics to $metrics_save_dir"

python scripts/compute_metrics_from_forecasts.py \
    recompute_metrics.eval_data_dir=$eval_data_dir \
    recompute_metrics.forecast_split=$forecast_split \
    recompute_metrics.labels_split=$labels_split \
    recompute_metrics.num_samples=null \
    recompute_metrics.context_length=512 \
    recompute_metrics.save_dir=$metrics_save_dir \
    "$@"