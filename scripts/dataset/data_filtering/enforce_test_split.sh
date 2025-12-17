#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 test_dir skew_train_dir skew_test_dir"
    exit 1
fi

test_dir=$1
skew_train_dir=$2
skew_test_dir=$3

# Create skew_test_dir if it doesn't exist
mkdir -p "$skew_test_dir"

# Iterate over each subdirectory in test_dir
for test_subdir in "$test_dir"/*; do
    if [ -d "$test_subdir" ]; then
        test_name=$(basename "$test_subdir")
        echo "test_name: $test_name"
        # Find matching subdirectories in skew_train_dir
        for skew_subdir in "$skew_train_dir"/*; do
            if [ -d "$skew_subdir" ]; then
                skew_name=$(basename "$skew_subdir")
                
                # Split skew_name into name1 and name2
                IFS='_' read -r name1 name2 <<< "$skew_name"
                
                # Check if test_name matches either name1 or name2
                if [[ "$test_name" == "$name1" || "$test_name" == "$name2" ]]; then
                    # Move the matching subdirectory to skew_test_dir
                    echo "found match: $skew_subdir"
                    mv "$skew_subdir" "$skew_test_dir"
                fi
            fi
        done
    fi
done
