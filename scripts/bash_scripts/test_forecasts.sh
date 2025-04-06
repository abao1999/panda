# run_num=387

# python tests/test_forecasts.py \
#     ChenLee_Chen \
#     HastingsPowell_SprottTorus \
#     Hopfield_DoubleGyre \
#     MacArthur_KawczynskiStrizhak \
#     --plot_save_dir figures/chronos_eval/forecasts/run-${run_num} \
#     --split_forecasts eval/chronos/run-${run_num}/forecasts \
#     --split_ground_truth eval/chronos/run-${run_num}/labels \

# run_num=388

# python tests/test_forecasts.py \
#     ChenLee_Chen \
#     HastingsPowell_SprottTorus \
#     Hopfield_DoubleGyre \
#     MacArthur_KawczynskiStrizhak \
#     --plot_save_dir figures/patchtst_eval/forecasts/run-${run_num} \
#     --split_forecasts eval/patchtst/run-${run_num}/forecasts \
#     --split_ground_truth eval/patchtst/run-${run_num}/labels \


run_name=pft_stand_rff_only_pretrained-0
split_dir=final_skew40/test_zeroshot

python tests/test_forecasts.py \
    all \
    --plot_save_dir figures/patchtst/${run_name}/${split_dir}/forecasts \
    --split_forecasts eval/patchtst/${run_name}/${split_dir}/forecasts \
    --split_ground_truth eval/patchtst/${run_name}/${split_dir}/labels \
    --num_systems 20


# run_name=chronos_finetune_stand_updated-0
# split_dir=final_skew40/test_zeroshot

# python tests/test_forecasts.py \
#     all \
#     --plot_save_dir figures/chronos/${run_name}/${split_dir}/forecasts \
#     --split_forecasts eval/chronos/${run_name}/${split_dir}/forecasts \
#     --split_ground_truth eval/chronos/${run_name}/${split_dir}/labels \
#     --num_systems 20
