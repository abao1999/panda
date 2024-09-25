#!/bin/bash

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <split> <dyst_name>"
    exit 1
fi

# Assign arguments to variables
split=$1
dyst_name=$2

# Construct the directory path
dir_to_remove="$WORK/data/$split/$dyst_name"

# Check if the directory exists
if [ ! -d "$dir_to_remove" ]; then
    echo "Error: Directory $dir_to_remove does not exist."
    exit 1
fi

# Ask for confirmation
echo "Are you sure you want to delete the directory: $dir_to_remove?"
echo "This action cannot be undone."
read -p "Type 'yes' to confirm: " confirmation

# Check the confirmation
if [ "$confirmation" = "yes" ]; then
    # Remove the directory
    rm -rf "$dir_to_remove"
    echo "Directory $dir_to_remove has been deleted."
else
    echo "Operation cancelled. No changes were made."
fi