#!/bin/bash

# skew systems
python -W ignore scripts/make_skew_systems.py \
    sampling.num_points=8192 \
    sampling.num_periods=40 \
    sampling.num_periods_min=20 \
    sampling.num_periods_max=60 \
    sampling.num_param_perturbations=100 \
    sampling.param_scale=1.0 \
    sampling.atol=1e-10 \
    sampling.rtol=1e-8 \
    sampling.silence_integration_errors=true \
    sampling.data_dir=/stor/work/AMDG_Gilpin_Summer2024/data/big_skew_mixedp \
    sampling.rseed=21434 \
    sampling.verbose=false \
    events.verbose=false \
    events.max_duration=600 \
    skew.normalization_strategy=flow_rms \
    skew.transform_scales=true \
    skew.randomize_driver_indices=true \
    skew.num_pairs=2048 \
    skew.pairs_rseed=123 \
    skew.sys_idx_low=0 \
    skew.sys_idx_high=null \
    run_name=big_skew_mixedp \

# # base dysts
# python -W ignore scripts/make_dyst_data.py \
#     sampling.sys_class=continuous \
#     sampling.num_points=8192 \
#     sampling.num_periods=40 \
#     sampling.num_periods_min=20 \
#     sampling.num_periods_max=60 \
#     sampling.num_param_perturbations=200 \
#     sampling.param_scale=1.0 \
#     sampling.atol=1e-10 \
#     sampling.rtol=1e-8 \
#     sampling.silence_integration_errors=true \
#     sampling.data_dir=/stor/work/AMDG_Gilpin_Summer2024/data/big_base_mixedp \
#     sampling.rseed=21433 \
#     sampling.verbose=false \
#     events.verbose=false \
#     events.max_duration=600 \
#     validator.transient_time_frac=0.5 \
#     run_name=big_base_mixedp \