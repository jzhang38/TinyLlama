export PYTHON_PATH="/root/TinyLlama:$PYTHON_PATH"
# python scripts/prepare_redpajama.py --source_path /root/data/RedPajama-Data-1T-Sample --checkpoint_dir llama-tokenizer  --destination_path /root/data/red_pajama_1_t_sample_tiny_llama --split train
# python scripts/prepare_starcoder.py --source_path /home/ken/workspace/data/tinyllama/starcoderdata --tokenizer_path llama-tokenizer --destination_path /media/ken/Data/slim_star_combined --split train --percentage 1.0
python scripts/prepare_slimpajama.py --source_path ../data/SlimPajama-627B --tokenizer_path llama-tokenizer --destination_path ../data/slim_star_combined --split train --percentage 1.0
python scripts/prepare_slimpajama.py --source_path /root/data/SlimPajama-627B --tokenizer_path llama-tokenizer --destination_path /root/data/slim_star_combined --split validation --percentage 1.0
