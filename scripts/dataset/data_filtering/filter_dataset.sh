#!/bin/bash

# Filter a dataset based on a JSON file that contains a list of subdirectories to reject.

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <data_directory> <json_file> <target_directory>"
    exit 1
fi

# Assign arguments to variables
DATA_DIR=$1
JSON_FILE=$2
TARGET_DIR=$3

# # Ask for confirmation before proceeding
# read -p "Do you want to proceed? (yes/no): " confirm
# if [ "$confirm" != "yes" ]; then
#     echo "Aborting..."
#     exit 1
# fi


# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "jq is required but not installed. Please install jq to use this script."
    exit 1
fi

# Read the JSON file and iterate over each key-value pair
jq -c 'to_entries[]' "$JSON_FILE" | while IFS= read -r line; do
    # Use jq to parse the line correctly
    subdir=$(echo "$line" | jq -r '.key')
    indices=$(echo "$line" | jq -r '.value | join(",")')
    echo "Processing subdir $subdir with indices $indices"
    
    # Get list of filepaths in ${DATA_DIR}/${subdir} and sort by numerical prefix
    filepaths=$(find "${DATA_DIR}/${subdir}" -type f | awk -F'/' '{print $NF}' | sort -t'_' -k1,1n)
    # echo "filepaths: $filepaths"

    # Iterate over each index
    for index in $(echo "$indices" | tr ',' ' '); do
        # !IMPORTANT: assumes that the Arrow files are named with a numeric prefix
        # (e.g., "1_T-1024.arrow"), but the index read from the json file is only
        # the index of the filepaths sorted by the prefix.
        echo "Processing index $index for subdir $subdir"
        # Construct the file pattern
        # file_pattern="${DATA_DIR}/${subdir}/${index}_T*"
        filename=$(echo "$filepaths" | awk -v idx="$((index+1))" 'NR==idx {print $0}')
        filepath="${DATA_DIR}/${subdir}/${filename}"
        echo "filename: $filename"
        
        # Move the files matching the pattern to the target directory
        if [ -e "$filepath" ]; then
            # Create the target subdirectory if it doesn't exist
            target_subdir="${TARGET_DIR}/${subdir}"
            mkdir -p "$target_subdir"
            
            # Move the file to the target subdirectory
            mv "$filepath" "$target_subdir"
            echo "Moved $filepath to $target_subdir"
        else
            echo "No file found matching $filepath"
        fi
    done
done
