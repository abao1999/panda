#!/bin/bash

filter_name_suffix="z5_z10"

data_dirs=("final_base20" "final_skew20" "final_base40" "final_skew40")
# data_dirs=("final_base20" "final_base40")
data_splits=("test_zeroshot" "test" "train")


for data_dir in "${data_dirs[@]}"; do
    find $WORK/data/${data_dir} -type d -empty -delete
    for data_split in "${data_splits[@]}"; do
        echo "Filtering ${data_dir}/${data_split}"

        python scripts/dataset_analysis.py \
            analysis.split=${data_dir}/${data_split} \
            analysis.filter_json_fname=failed_samples_${filter_name_suffix} \
            analysis.check_boundedness.max_zscore=5

        echo "Writing output json to outputs/${data_dir}/${data_split}/failed_samples_${filter_name_suffix}.json"

        ./scripts/bash_scripts/filter_dataset.sh \
            $WORK/data/${data_dir}/${data_split} \
            outputs/${data_dir}/${data_split}/failed_samples_${filter_name_suffix}.json \
            $WORK/data/${data_dir}/${data_split}_${filter_name_suffix}

        find $WORK/data/${data_dir}/${data_split} -type d -empty -delete

    done
done
