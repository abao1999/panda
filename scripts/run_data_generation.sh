#!/bin/bash

# python -W ignore scripts/make_dyst_data.py \
#     dyst_data.sys_class=continuous \
#     dyst_data.split_prefix=debug \
#     dyst_data.num_ics=1 \
#     dyst_data.num_param_perturbations=1 \
#     dyst_data.param_scale=0.5 \
#     events.max_duration=120 \
#     events.min_step=1e-18 \
#     validator.plot_save_dir=tests/plots \

# couple flow space
python -W ignore scripts/make_skew_systems_old.py \
    dyst_data.sys_class=continuous_no_delay \
    dyst_data.split_prefix=flow_run2 \
    dyst_data.num_param_perturbations=50 \
    dyst_data.param_scale=1.0 \
    dyst_data.rseed=123 \
    skew.couple_phase_space=False \
    skew.couple_flows=True \
    skew.n_combos=1040 \
    events.max_duration=300 \
