# lightning run model \
#     --node-rank=0  \
#     --accelerator=cuda \
#     --devices=1 \
#     --num-nodes=1 \
#     pretrain/tinyllama.py --devices 1 --train_data_dir /media/ken/Data/slim_star_combined  --val_data_dir /home/ken/workspace/TinyLlama/data/slim_star_combined
    
# pretrain/tinyllama.py --devices 1 --train_data_dir /home/ken/workspace/TinyLlama/data/slim_star_combined  --val_data_dir /home/ken/workspace/TinyLlama/data/slim_star_combined

wandb login

# For validation
DATASET=/root/data/red_pajama_1_t_sample_tiny_llama
DEVICES=4
lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=$DEVICES \
    --num-nodes=1 \
    pretrain/tinyllama.py --devices $DEVICES --train_data_dir $DATASET  --val_data_dir $DATASET
