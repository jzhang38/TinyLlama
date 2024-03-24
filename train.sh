# lightning run model \
#     --node-rank=0  \
#     --accelerator=cuda \
#     --devices=1 \
#     --num-nodes=1 \
#     pretrain/tinyllama.py --devices 1 --train_data_dir /media/ken/Data/slim_star_combined  --val_data_dir /home/ken/workspace/TinyLlama/data/slim_star_combined
    
# pretrain/tinyllama.py --devices 1 --train_data_dir /home/ken/workspace/TinyLlama/data/slim_star_combined  --val_data_dir /home/ken/workspace/TinyLlama/data/slim_star_combined

lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=1 \
    --num-nodes=1 \
    pretrain/tinyllama.py --devices 1 --train_data_dir /media/ken/Data/red_pajama_1_t_sample_tiny_llama  --val_data_dir /media/ken/Data/red_pajama_1_t_sample_tiny_llama