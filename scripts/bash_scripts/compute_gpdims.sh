
ulimit -n 99999

test_data_dirs=(
    $WORK/data/improved/final_base40/test_zeroshot
    $WORK/data/improved/final_skew40/test_zeroshot
)
test_data_dirs_json=$(printf '%s\n' "${test_data_dirs[@]}" | jq -R . | jq -s -c .)
echo "test_data_dirs: $test_data_dirs_json"

# run_name=mlm_stand_chattn_noembed-0
run_name=panda_mlm_nh12_dmodel768_mixedp-2

python scripts/compute_gpdims.py \
    eval.mode=pretrain \
    eval.checkpoint_path=$WORK/checkpoints/$run_name/checkpoint-final \
    eval.device=cuda:5 \
    eval.torch_dtype=float16 \
    eval.data_paths_lst=$test_data_dirs_json \
    eval.num_subdirs=null \
    eval.num_samples_per_subdir=null \
    eval.metrics_save_dir=$WORK/eval_results/patchtst/$run_name/test_zeroshot \
    eval.metrics_fname=ngpdims \
    eval.save_completions=false \
    eval.num_processes=100 \
    eval.completions.start_time=512 \
    eval.completions.end_time=null