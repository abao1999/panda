#!/bin/bash

data_dir=$WORK/data/improved/final_base40
split_name=train

params_json_path=$data_dir/parameters/${split_name}/more_extra.json

if [ ! -f "$params_json_path" ]; then
    echo "Error: Parameter file does not exist: $params_json_path"
    echo "Skipping this split"
    exit 1
fi
echo "Parameter file exists: $params_json_path"

# directory to save the new dataset
data_split_dir_more_ics=$data_dir/${split_name}_more_ics

n_ics=16

python scripts/make_dataset_from_params.py \
    restart_sampling.split_name=$split_name \
    restart_sampling.params_json_path=$params_json_path \
    restart_sampling.systems_batch_size=128 \
    restart_sampling.batch_idx_low=0 \
    restart_sampling.batch_idx_high=null \
    sampling.data_dir=$data_split_dir_more_ics \
    sampling.rseed=6411 \
    sampling.num_ics=$n_ics \
    sampling.num_points=5120 \
    sampling.num_periods=40 \
    sampling.num_periods_min=25 \
    sampling.num_periods_max=100 \
    sampling.split_coords=false \
    sampling.atol=1e-10 \
    sampling.rtol=1e-8 \
    sampling.silence_integration_errors=true \
    events.verbose=false \
    events.max_duration=400 \
    validator.transient_time_frac=0.2 \
    wandb.log=false \
    wandb.project_name=dyst_data \
    "$@"
