#!/bin/bash

# Define source and destination directories
# Move directories from dirB to dirA
dirA="$WORK/data/improved/final_base40/test_zeroshot"
dirB="$WORK/data/improved/final_base40/test_zeroshot_z5_z10"
echo "moving from $dirB to $dirA"

# Ensure dirB exists
if [ ! -d "$dirB" ]; then
    echo "Source directory $dirB does not exist."
    exit 1
fi

# Iterate over subdirectories in dirB
for subdir in "$dirB"/*; do
    if [ -d "$subdir" ]; then
        subdir_name=$(basename "$subdir")
        target_dir="$dirA/$subdir_name"

        # Ensure target subdirectory exists in dirA
        mkdir -p "$target_dir"

        # Move files from dirB/subdirX to dirA/subdirX
        mv "$subdir"/* "$target_dir" 2>/dev/null

        # Remove the now-empty directory in dirB
        rmdir "$subdir" 2>/dev/null
    fi
done
