#!/bin/bash

# # skew systems
# python -W ignore scripts/make_skew_systems.py \
#     sampling.num_points=4311 \
#     sampling.num_periods=80 \
#     sampling.num_param_perturbations=100 \
#     sampling.param_scale=1.0 \
#     sampling.atol=1e-10 \
#     sampling.rtol=1e-8 \
#     sampling.silence_integration_errors=true \
#     sampling.data_dir=/stor/work/AMDG_Gilpin_Summer2024/data/big_skew80_run1 \
#     sampling.rseed=14916 \
#     sampling.verbose=false \
#     events.verbose=false \
#     events.max_duration=300 \
#     skew.normalization_strategy=flow_rms \
#     skew.transform_scales=true \
#     skew.randomize_driver_indices=true \
#     skew.num_pairs=2048 \
#     run_name=big_skew80_run1 \

# base dysts
python -W ignore scripts/make_dyst_data.py \
    sampling.sys_class=continuous \
    sampling.num_points=4311 \
    sampling.num_periods=80 \
    sampling.num_param_perturbations=200 \
    sampling.param_scale=1.0 \
    sampling.atol=1e-10 \
    sampling.rtol=1e-8 \
    sampling.silence_integration_errors=true \
    sampling.data_dir=/stor/work/AMDG_Gilpin_Summer2024/data/big_base80_run1 \
    sampling.rseed=11217 \
    sampling.verbose=false \
    events.verbose=false \
    events.max_duration=300 \
    run_name=big_base80_run1 \