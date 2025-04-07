#!/bin/bash
# Usage: ./move_samples.sh /path/to/source /path/to/destination

SRC_DIR="$1"
DEST_DIR="$2"
n_files_to_move="$3"

if [[ -z "$SRC_DIR" || -z "$DEST_DIR" ]]; then
  echo "Usage: $0 <source_directory> <destination_directory>"
  exit 1
fi

n_overlap_subdirs=0
# Iterate over each subdirectory in the source directory
for subdir in "$SRC_DIR"/*/ ; do
  # Ensure it's a directory
  if [ -d "$subdir" ]; then
    subdirName=$(basename "$subdir")

    echo "Processing $subdirName"
    
    # Create the corresponding subdirectory in the destination directory
    if [ -d "$DEST_DIR/$subdirName" ]; then
      echo "Warning: Directory $DEST_DIR/$subdirName already exists"
      n_overlap_subdirs=$((n_overlap_subdirs + 1))
    fi
    mkdir -p "$DEST_DIR/$subdirName"
    
    # Find files matching the pattern and sort by the numeric prefix before the underscore.
    # Then select the first two files.
    files=$(ls "$subdir"*"_T-4096.arrow" 2>/dev/null | sort -V | head -n "$n_files_to_move")
    # echo "Found:"
    # echo "$files"

    # Move each selected file into the destination subdirectory.
    for file in $files; do
    #   echo "Moving $file"
      if [ -f "$file" ]; then
        cp "$file" "$DEST_DIR/$subdirName/"
      fi
    done
  fi
done

echo "Move completed."
echo "Number of overlapping subdirectories: $n_overlap_subdirs"