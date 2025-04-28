#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints

ulimit -n 99999

# scaling law runs
run_names=(
    pft_chattn_mlm_sys10490_ic2-0
    pft_chattn_mlm_sys656_ic32-0
    pft_chattn_mlm_sys164_ic128-0
    pft_chattn_mlm_sys5245_ic4-0
)

# # univariate with old dynamics embedding
# run_names=(
#     pft_emb_equal_param_univariate_from_scratch-0
#     pft_rff_univariate_pretrained-0
# )

# # univariate either without dynamics embedding or with the new poly one
# run_names=(
#     pft_noemb_equal_param_univariate_from_scratch-0
#     pft_vanilla_pretrained_correct-0
#     pft_equal_param_deeper_univariate_from_scratch_noemb-0
# )

# # multivariate with old dynamics embedding
# run_names=(
#     pft_stand_rff_only_pretrained-0 
#     pft_fullyfeat_from_scratch-0 # this is actually just rff from scratch
# )

# # multivariate either without dynamics embedding or with the new poly one
# run_names=(
#     # pft_chattn_noembed_pretrained_correct-0 
#     pft_stand_chattn_noemb-0 
#     pft_chattn_fullemb_quartic_enc-0
#     pft_chattn_emb_w_poly-0
#     pft_chattn_fullemb_pretrained-0
# )

# split_dir=final_skew40/train
split_dir=final_skew40/test_zeroshot
model_dirname=patchtst

for run_name in ${run_names[@]}; do
    echo "Evaluating $run_name"
    python scripts/patchtst/evaluate.py \
        eval.mode=predict \
        eval.checkpoint_path=$checkpoint_dir/$run_name/checkpoint-final \
        eval.data_path=$WORK/data/improved/$split_dir \
        eval.num_systems=null \
        eval.num_samples_per_subdir=null \
        eval.num_test_instances=6 \
        eval.window_style=sampled \
        eval.batch_size=64 \
        eval.context_length=512 \
        eval.prediction_length=512 \
        eval.limit_prediction_length=false \
        eval.metrics_save_dir=$WORK/eval_results/$model_dirname/$run_name/$split_dir \
        eval.metrics_fname=metrics \
        eval.overwrite=true \
        eval.device=cuda:3 \
        eval.save_predictions=false \
        eval.save_labels=false \
        eval.forecast_save_dir=$WORK/data/eval/$model_dirname/$run_name/$split_dir/forecasts \
        eval.labels_save_dir=$WORK/data/eval/$model_dirname/$run_name/$split_dir/labels \
        eval.seed=99 \
        "$@"
done