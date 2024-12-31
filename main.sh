#!/bin/bash
# Array of main directory names
declare -a main_dirs=("D1+D2" "D1+D3" "D2+D3" "D1+D2+D3")

# obtain the actual path
base_path=$(dirname "$(realpath $0)")

for dir in "${main_dirs[@]}"; do
  # Create the main directory
  mkdir -p "$base_path/$dir"
  # Create subdirectories within the main directory
  mkdir -p "$base_path/$dir/p1_output"
  mkdir -p "$base_path/$dir/p2_graphs"
  mkdir -p "$base_path/$dir/p2_output"

done

declare -a direct=("D1" "D2" "D3")

for dir in "${direct[@]}"; do
  mkdir -p "$base_path/$dir/p1_output"
  mkdir -p "$base_path/$dir/p2_graphs"
  mkdir -p "$base_path/$dir/p2_output"
done

echo "Directories and subdirectories have been created."
