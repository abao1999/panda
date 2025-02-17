# run_num=387

# python tests/test_forecasts.py \
#     ChenLee_Chen \
#     HastingsPowell_SprottTorus \
#     Hopfield_DoubleGyre \
#     MacArthur_KawczynskiStrizhak \
#     --plot_save_dir figs/chronos_eval/forecasts/run-${run_num} \
#     --split_forecasts eval/chronos/run-${run_num}/forecasts \
#     --split_ground_truth eval/chronos/run-${run_num}/labels \

# run_num=388

# python tests/test_forecasts.py \
#     ChenLee_Chen \
#     HastingsPowell_SprottTorus \
#     Hopfield_DoubleGyre \
#     MacArthur_KawczynskiStrizhak \
#     --plot_save_dir figs/patchtst_eval/forecasts/run-${run_num} \
#     --split_forecasts eval/patchtst/run-${run_num}/forecasts \
#     --split_ground_truth eval/patchtst/run-${run_num}/labels \


run_num=380

python tests/test_forecasts.py \
    all \
    --plot_save_dir figs/patchtst_eval_debug/forecasts/run-${run_num} \
    --split_forecasts eval_debug/patchtst/run-${run_num}/forecasts \
    --split_ground_truth eval_debug/patchtst/run-${run_num}/labels \


# run_num=387

# python tests/test_forecasts.py \
#     all \
#     --plot_save_dir figs/chronos_eval_debug/forecasts/run-${run_num} \
#     --split_forecasts eval/chronos/run-${run_num}/forecasts \
#     --split_ground_truth eval/chronos/run-${run_num}/labels \