#!/bin/bash

# Usage: ./move_by_index.sh dirA dirB

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_dirA> <destination_dirB>"
    exit 1
fi

src_root="$1"
dst_root="$2"

# Ensure destination root exists
mkdir -p "$dst_root"

# Loop over subdirectories in dirA
find "$src_root" -mindepth 1 -maxdepth 1 -type d | while read -r subdir; do
    subname=$(basename "$subdir")
    mkdir -p "$dst_root/$subname"

    echo "Processing $subname"

    for file in "$subdir"/pp*_*_T-4096.arrow; do
        # Skip if no matching files
        [ -e "$file" ] || continue

        fname=$(basename "$file")
        if [[ "$fname" =~ ^pp[^_]+_([0-9]+)_T-4096\.arrow$ ]]; then
            idx="${BASH_REMATCH[1]}"
            if (( idx >= 8 )); then
                # echo "Moving $file to $dst_root/$subname/"
                mv "$file" "$dst_root/$subname/"
            fi
        fi
    done
done
