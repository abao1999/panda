#! /bin/bash

run_names_lst=(
    pft_chattn_fullemb_pretrained-0
    pft_chattn_noembed_pretrained_correct-0
    pft_chattn_emb_w_poly-0
    pft_stand_chattn_noemb-0
    pft_noemb_equal_param_univariate_from_scratch-0
    pft_equal_param_deeper_univariate_from_scratch_noemb-0
)
split=final_skew40/test_zeroshot

model_dirname=patchtst
use_sliding_context=true

if [ "$use_sliding_context" = true ]; then
    model_dirname=patchtst_sliding
    echo "Using sliding context"
fi

for run_name in "${run_names_lst[@]}"; do
    eval_data_dir=$WORK/data/eval/${model_dirname}/${run_name}/${split}

    echo "eval_data_dir: $eval_data_dir"

    forecast_split="forecasts"
    labels_split="labels"

    echo "eval_data_dir: $eval_data_dir"
    echo "forecast_split: $forecast_split"
    echo "labels_split: $labels_split"

    metrics_save_dir=$WORK/recomputed_eval_results/${model_dirname}/${run_name}/${split}
    echo "Saving metrics to $metrics_save_dir"

    python scripts/compute_metrics_from_forecasts.py \
        recompute_metrics.eval_data_dir=$eval_data_dir \
        recompute_metrics.forecast_split=$forecast_split \
        recompute_metrics.labels_split=$labels_split \
        recompute_metrics.num_systems=null \
        recompute_metrics.context_length=512 \
        recompute_metrics.save_dir=$metrics_save_dir \
        recompute_metrics.metrics_fname=metrics \
        "$@"
done
