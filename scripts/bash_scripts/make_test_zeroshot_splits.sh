#!/bin/bash

num_periods=(40)

for period in "${num_periods[@]}"; do
    echo "period: $period"
    base_dir=final_base${period}_fixed
    skew_dir=final_skew${period}_fixed

    ./scripts/bash_scripts/enforce_test_split.sh \
        $WORK/data/${base_dir}/test_zeroshot \
        $WORK/data/${skew_dir}/train \
        $WORK/data/${skew_dir}/test_zeroshot
done