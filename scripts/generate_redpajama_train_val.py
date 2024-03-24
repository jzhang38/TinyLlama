import os
import glob
import random

# Specify the folder to search in
folder_path = '/media/ken/Data/red_pajama_1_t_sample_tiny_llama/'

# Find all .bin files in the folder
bin_files = glob.glob(os.path.join(folder_path, '*.bin'))

# Group files by their prefix (index 2 when split by '_')
# e.g. train_redpajamav1_wikipedia_sample_10_0000000010.bin -> wikipedia
grouped_files = {}
for file_path in bin_files:
    parts = os.path.basename(file_path).split('_')
    if len(parts) > 2:
        prefix = parts[2]  # Extract the third element as prefix
        if prefix in grouped_files:
            grouped_files[prefix].append(file_path)
        else:
            grouped_files[prefix] = [file_path]

# Process each group
for prefix, files in grouped_files.items():
    # Calculate 10% of the files for validation
    num_files_to_validate = max(1, len(files) // 10)
    
    # Shuffle the files to randomly pick
    random.shuffle(files)
    
    # Select files for validation and for training
    validation_files = files[:num_files_to_validate]
    training_files = files[num_files_to_validate:]
    
    # Rename validation files
    for file_path in validation_files:
        dir_name, file_name = os.path.split(file_path)
        new_file_name = file_name.replace("train_", "validation_")
        new_file_path = os.path.join(dir_name, new_file_name)
        os.rename(file_path, new_file_path)
        print(f'Renamed to "{new_file_name}" for validation.')

    # Note: For training files, we're only adding the prefix if it's not already present
    for file_path in training_files:
        dir_name, file_name = os.path.split(file_path)
        if not file_name.startswith('train_'):
            new_file_name = 'train_' + file_name
            new_file_path = os.path.join(dir_name, new_file_name)
            os.rename(file_path, new_file_path)
            print(f'Added prefix to "{new_file_name}" for training.')

print(f'Processing completed.')