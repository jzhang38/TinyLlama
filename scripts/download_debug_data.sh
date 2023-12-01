#!/bin/bash

# Base URL for the files
mkdir data
cd data
mkdir SlimPajama-627B
cd SlimPajama-627B
mkdir train && cd train
mkdir chunk1 && cd chunk1
base_url_train="https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/train/chunk1/"
base_url_val="https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/train/chunk2/"
# Start and end file numbers

# Loop to download each file
for (( i=0; i<=1000; i++ ))
do
    # Construct the file name
    file_name="example_train_${i}.jsonl.zst"

    # Full URL
    full_url="${base_url_train}${file_name}"

    # Download the file using wget
    wget "$full_url"
done

cd ../..
mkdir validation && cd validation
mkdir chunk1 && cd chunk1
# Loop to download each file
for (( i=0; i<=50; i++ ))
do
    # Construct the file name
    file_name="example_train_${i}.jsonl.zst"

    # Full URL
    full_url="${base_url_val}${file_name}"
    # Download the file using wget
    wget "$full_url"
done
echo "Download completed."

wget 

