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