export PYTHON_PATH="/home/ken/workspace/TinyLlama:$PYTHON_PATH"
python scripts/prepare_redpajama.py --source_path /media/ken/Data/RedPajama-Data-1T-Sample --checkpoint_dir llama-tokenizer  --destination_path /media/ken/Data/red_pajama_1_t_sample_tiny_llama --split train
# python scripts/prepare_starcoder.py --source_path /home/ken/workspace/data/tinyllama/starcoderdata --tokenizer_path llama-tokenizer --destination_path /media/ken/Data/slim_star_combined --split train --percentage 1.0
