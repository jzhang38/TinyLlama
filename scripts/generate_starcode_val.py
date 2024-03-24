import os
import glob
import random

# Specify the folder to search in
folder_path = '/media/ken/Data/slim_star_combined/'

# Find all .bin files with 'train_' prefix in the folder
bin_files = glob.glob(os.path.join(folder_path, 'train_*.bin'))

# Calculate 10% of the found files
num_files_to_select = max(1, len(bin_files) // 10)

# Randomly select 10% of the files
selected_files = random.sample(bin_files, num_files_to_select)

# Rename the selected files by changing their prefix from 'train_' to 'validation_'
for file_path in selected_files:
    # Split the directory and filename
    dir_name, file_name = os.path.split(file_path)
    
    # Check and change the prefix
    if file_name.startswith('train_'):
        new_file_name = 'validation_' + file_name[len('train_'):]
        new_file_path = os.path.join(dir_name, new_file_name)
        
        # Rename the file
        os.rename(file_path, new_file_path)
        print(f'Renamed "{file_path}" to "{new_file_path}"')

print(f'Total files renamed: {len(selected_files)}')
