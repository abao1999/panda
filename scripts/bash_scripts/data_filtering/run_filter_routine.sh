#!/bin/bash

# filter_name_suffix="z5_z10"
filter_name_suffix="zero_one_test"

# data_dirs=("final_base20" "final_skew20" "final_base40" "final_skew40")
# data_dirs=("final_base20" "final_base40")
# data_splits=("test_zeroshot" "test" "train")

data_dirs=("final_skew80" "final_base80")
data_splits=("test_zeroshot" "test_zeroshot_z5_z10" "test_zeroshot_z10_z15" "train" "train_z5_z10" "train_z10_z15")


for data_dir in "${data_dirs[@]}"; do
    find $WORK/data/${data_dir} -type d -empty -delete
    for data_split in "${data_splits[@]}"; do
        echo "Filtering ${data_dir}/${data_split}"

        # python scripts/dataset_analysis.py \
        #     analysis.split=${data_dir}/${data_split} \
        #     analysis.num_samples=null \
        #     analysis.filter_ensemble=true \
        #     analysis.attractor_tests='["check_boundedness"]' \
        #     analysis.filter_json_fname=failed_samples_${filter_name_suffix} \

        python scripts/dataset_analysis.py \
            analysis.split=${data_dir}/${data_split} \
            analysis.num_samples=null \
            analysis.filter_ensemble=true \
            analysis.attractor_tests='["check_zero_one_test"]' \
            analysis.filter_json_fname=failed_samples_${filter_name_suffix} \
            analysis.check_zero_one_test.threshold=0.2 \
            analysis.check_zero_one_test.strategy=score

        echo "Writing output json to outputs/${data_dir}/${data_split}/failed_samples_${filter_name_suffix}.json"

        ./scripts/bash_scripts/filter_dataset.sh \
            $WORK/data/${data_dir}/${data_split} \
            outputs/${data_dir}/${data_split}/failed_samples_${filter_name_suffix}.json \
            $WORK/data/${data_dir}/${data_split}_${filter_name_suffix}

        find $WORK/data/${data_dir}/${data_split} -type d -empty -delete

    done
done
