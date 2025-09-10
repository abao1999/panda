#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints

ulimit -n 99999

# scaling law runs
run_names_scalinglaw=(
    pft_chattn_mlm_sys10490_ic2-0
    pft_chattn_mlm_sys656_ic32-0
    pft_chattn_mlm_sys164_ic128-0
    pft_chattn_mlm_sys5245_ic4-0
    pft_chattn_mlm_sys1312_ic16-0
    pft_chattn_mlm_sys328_ic64-0
    pft_chattn_mlm_sys2623_ic8-0
    pft_chattn_mlm_sys20k_ic1-0
)

# univariate with old dynamics embedding
run_names_univariate_kernelemb_old=(
    pft_emb_equal_param_univariate_from_scratch-0
    pft_rff_univariate_pretrained-0
)

# univariate either without dynamics embedding or with the new poly one
run_names_univariate=(
    pft_noemb_equal_param_univariate_from_scratch-0
    pft_vanilla_pretrained_correct-0
    pft_equal_param_deeper_univariate_from_scratch_noemb-0
)

# multivariate without dynamics embedding
run_names_multivariate=(
    pft_chattn_noembed_pretrained_correct-0 
    pft_stand_chattn_noemb-0 
    pft_chattn_noemb_pretrained_chrope-0
)

# multivariate with the kernel embedding
run_names_multivariate_kernelemb=(
    pft_rff496_proj-0
    pft_chattn_emb_w_poly-0
    pft_chattn_fullemb_pretrained-0
)

# multivariate with linear attention polyfeats dynamics embedding
run_names_multivariate_linattnpolyemb=(
    pft_linattnpolyemb_from_scratch-0
)

run_names_new=(
    pft_sft-0
)

run_names_patch_ablations=(
    panda21M_ps4-0
    panda21M_ps32-0
    panda21M_ps8-1
)

run_names=(
    # panda_nh12_dmodel768_mixedp-4
    # panda_nh10_dmodel640-1
    ${run_names_patch_ablations[@]}
    # ${run_names_new[@]}
    # ${run_names_scalinglaw[@]}
    # ${run_names_univariate[@]}
    # ${run_names_univariate_kernelemb_old[@]}
    # ${run_names_multivariate[@]}
    # ${run_names_multivariate_kernelemb_old[@]}
    # ${run_names_multivariate_kernelemb[@]}
    # ${run_names_multivariate_linattnpolyemb[@]}
)

echo "run_names: ${run_names[@]}"

test_data_dirs=(
    $WORK/data/improved/final_base40/test_zeroshot
    $WORK/data/improved/final_base40/test_zeroshot_z5_z10
    $WORK/data/improved/final_skew40/test_zeroshot
    $WORK/data/improved/final_skew40/test_zeroshot_z5_z10
)
test_data_dirs_json=$(printf '%s\n' "${test_data_dirs[@]}" | jq -R . | jq -s -c .)
echo "test_data_dirs: $test_data_dirs_json"

for run_name in ${run_names[@]}; do
    echo "Evaluating $run_name"
    python scripts/patchtst/evaluate.py \
        eval.mode=predict \
        eval.checkpoint_path=$checkpoint_dir/$run_name/checkpoint-final \
        eval.data_paths_lst=$test_data_dirs_json \
        eval.num_subdirs=null \
        eval.num_test_instances=6 \
        eval.window_style=sampled \
        eval.batch_size=64 \
        eval.context_length=512 \
        eval.prediction_length=512 \
        eval.limit_prediction_length=false \
        eval.metrics_save_dir=$WORK/eval_results/patchtst/$run_name/test_zeroshot \
        eval.metrics_fname=metrics \
        eval.overwrite=true \
        eval.device=cuda:2 \
        eval.save_labels=false \
        eval.save_forecasts=false
done