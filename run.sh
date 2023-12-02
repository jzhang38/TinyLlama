#!/bin/bash

# Loop through all yaml files with the prefix 'llama_19M' in the experiments directory
for file in experiments/llama_19M*.yaml; do
    # Run the python command with each file
    python pretrain/tinyllama.py --training_config "$file"
done