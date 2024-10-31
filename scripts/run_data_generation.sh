main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir=$WORK/checkpoints



# Couple phase space: NOTE: need to be very careful beccause x+y may not be valid parameter choice for response
python -W ignore scripts/make_skew_systems.py all \
    --split-prefix=big \
    --couple_phase_space=True \
    --couple_flows=False \
    --n_combos=128 \
    --max-duration=240 \
    --num-param-perturbations=10 \
    --param-scale=0.5 \
    --sys-class=continuous_no_delay \


# couple flow space
python -W ignore scripts/make_skew_systems.py all \
    --split-prefix=big_flow \
    --couple_phase_space=False \
    --couple_flows=True \
    --n_combos=128 \
    --max-duration=240 \
    --num-param-perturbations=10 \
    --param-scale=0.5 \
    --sys-class=continuous_no_delay \