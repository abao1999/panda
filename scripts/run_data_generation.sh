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

# # Couple phase space: NOTE: need to be very careful beccause x+y may not be valid parameter choice for response
# python -W ignore scripts/make_skew_systems.py \
#     dyst_data.sys_class=continuous_no_delay \
#     dyst_data.split_prefix=big \
#     dyst_data.num_param_perturbations=10 \
#     dyst_data.param_scale=0.5 \
#     skew.couple_phase_space=True \
#     skew.couple_flows=False \
#     skew.n_combos=100 \
#     events.max_duration=240 \

# # couple flow space
# python -W ignore scripts/make_skew_systems.py \
#     dyst_data.sys_class=continuous_no_delay \
#     dyst_data.split_prefix=big_flow \
#     dyst_data.num_param_perturbations=20 \
#     dyst_data.param_scale=0.5 \
#     skew.couple_phase_space=False \
#     skew.couple_flows=True \
#     skew.n_combos=1000 \
#     events.max_duration=300 \


# relatively small run for debugging
python -W ignore scripts/make_skew_systems.py \
    dyst_data.sys_class=continuous_no_delay \
    dyst_data.split_prefix=big_flow \
    dyst_data.num_param_perturbations=10 \
    dyst_data.param_scale=0.5 \
    skew.couple_phase_space=False \
    skew.couple_flows=True \
    skew.n_combos=128 \
    events.max_duration=180 \