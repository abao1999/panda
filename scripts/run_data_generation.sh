#!/bin/bash

python -W ignore scripts/make_dyst_data.py \
    sampling.sys_class=continuous \
    sampling.num_points=4096 \
    sampling.num_periods=10 \
    sampling.num_param_perturbations=200 \
    sampling.param_scale=1.0 \
    sampling.atol=1e-10 \
    sampling.rtol=1e-8 \
    sampling.silence_integration_errors=true \
    sampling.split_prefix=new_base_run4 \
    sampling.rseed=43

# python -W ignore scripts/make_skew_systems.py \
#     sampling.num_points=4096 \
#     sampling.num_periods=10 \
#     sampling.num_param_perturbations=20 \
#     sampling.atol=1e-10 \
#     sampling.rtol=1e-8 \
#     sampling.silence_integration_errors=true \
#     sampling.split_prefix=debug_run1 \
#     skew.normalization_strategy=mean_amp_response \
#     skew.transform_scales=false \
#     skew.randomize_driver_indices=false \
#     skew.num_pairs=200 \

