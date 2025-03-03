#!/bin/bash

num_periods=(80)

for period in "${num_periods[@]}"; do
    echo "period: $period"
    base_dir=final_base${period}
    skew_dir=final_skew${period}

    ./scripts/bash_scripts/enforce_test_split.sh \
        $WORK/data/${base_dir}/test_zeroshot \
        $WORK/data/${skew_dir}/train \
        $WORK/data/${skew_dir}/test_zeroshot
done