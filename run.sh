#!/bin/bash

# Loop through all yaml files with the prefix 'llama' in the experiments directory
for file in experiments/node1/llama*.yaml; do
    # Run the python command with each file
    python pretrain/tinyllama.py --training_config "$file"
done

for file in experiments/node2/llama*.yaml; do
    # Run the python command with each file
    python pretrain/tinyllama.py --training_config "$file"
done

for file in experiments/node3/llama*.yaml; do
    # Run the python command with each file
    python pretrain/tinyllama.py --training_config "$file"
done

for file in experiments/node4/llama*.yaml; do
    # Run the python command with each file
    python pretrain/tinyllama.py --training_config "$file"
done

for file in experiments/node5/llama*.yaml; do
    # Run the python command with each file
    python pretrain/tinyllama.py --training_config "$file"
done

for file in experiments/node6/llama*.yaml; do
    # Run the python command with each file
    python pretrain/tinyllama.py --training_config "$file"
done


for file in experiments/node7/llama*.yaml; do
    # Run the python command with each file
    python pretrain/tinyllama.py --training_config "$file"
done


for file in experiments/node8/llama*.yaml; do
    # Run the python command with each file
    python pretrain/tinyllama.py --training_config "$file"
done




for file in llama_*.yaml; do
    # Check if the file name contains "_linear_logx_"
    if [[ $file == *"_step_"* ]]; then
        # Replace "_linear_logx_" with "_powerlaw_0.5_" and store the new name
        new_name="${file/_step/_linear_step}"
        
        # Rename the file
        mv "$file" "$new_name"
    fi
done