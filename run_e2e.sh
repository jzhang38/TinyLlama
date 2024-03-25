# echo "Setting up wandb"
# pip install wandb
# wandb login

# echo "Install dependencies in background"
# ./setup.sh &> setup.log &

# echo "Start data download"
# pushd /root
# mkdir data
# cd data
# git lfs install
# git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
# popd

# Git clone process id
PID=514

# Check if the process is running by sending signal 0
# Loop until the process is not found
while kill -0 $PID 2>/dev/null; do
    echo "Data download process $PID is still running (checking every minute)."
    sleep 60
done

echo "Data download process $PID has finished."

echo "Processing data"
export PYTHON_PATH="/root/TinyLlama:$PYTHON_PATH"
DATASET=/root/data/slim_star_combined
python scripts/prepare_slimpajama.py --source_path /root/data/SlimPajama-627B --tokenizer_path llama-tokenizer --destination_path $DATASET --split train --percentage 1.0
python scripts/prepare_slimpajama.py --source_path /root/data/SlimPajama-627B --tokenizer_path llama-tokenizer --destination_path $DATASET --split validation --percentage 1.0

echo "Kill small validation run if needed"
if ps -p 6860 > /dev/null; then
    echo "Process 6860 is running. Killing it..."
    # Kill process 6860
    kill -9 6860
    echo "Process 6860 killed."
else

echo "Start training"
DEVICES=4
lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=$DEVICES \
    --num-nodes=1 \
    pretrain/tinyllama.py --devices $DEVICES --train_data_dir $DATASET  --val_data_dir $DATASET
