#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <data_directory> <json_file> <target_directory>"
    exit 1
fi

# Assign arguments to variables
DATA_DIR=$1
JSON_FILE=$2
TARGET_DIR=$3

# Ask for confirmation before proceeding
read -p "Do you want to proceed? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborting..."
    exit 1
fi


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
    
    # Iterate over each index
    for index in $(echo "$indices" | tr ',' ' '); do
        echo "Processing index $index for subdir $subdir"
        # Construct the file pattern
        file_pattern="${DATA_DIR}/${subdir}/${index}_T*"
        
        # Move the files matching the pattern to the target directory
        for file in $file_pattern; do
            if [ -e "$file" ]; then
                # Create the target subdirectory if it doesn't exist
                target_subdir="${TARGET_DIR}/${subdir}"
                mkdir -p "$target_subdir"
                
                # Move the file to the target subdirectory
                mv "$file" "$target_subdir"
                echo "Moved $file to $target_subdir"
            else
                echo "No files found for pattern $file_pattern"
            fi
        done
    done
done