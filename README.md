## Pretrain TinyLlama

### Installation
We expect you have CUDA 11.8 installed.

#### Install Pytorch and Xformers
```bash
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
```
(Xformers will automatically install PyTorch)

#### Install Flash-Attention 2 and other fused operators:
```bash
pip uninstall ninja -y && pip install ninja -U
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
python setup.py install
cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../.. && rm -rf flash-attention
```
#### Install Remaining Dependencies
```
pip install -r requirements.txt 
```
to install other dependencies.
It may take >= 5 minutes to build xformers/flash-attention. Do not worry if the process seemly stagnant or the terminal print out many warnings.

Then you are ready to go 🎉!

### Data Preparation
You need 1T disk space to download data and reproduce all our experiments.
#### Download Datasets
Download the Slimpajama to your chosen directory. (Note that to save space, we only use chunk 1 to chunk 4 in the training set for our experiments)
```bash
mkdir data
cd data
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
cd SlimPajama-627B
rm -rf .git
cd train
rm chunk5 chunk6 chunk7 chunk8 chunk9 chunk10
cd ../../..
```

#### Tokenize data
Use the provided scripts to tokenize the datasets and divide them into chunks.
```bash
python scripts/prepare_slimpajama.py --source_path data/SlimPajama-627B --tokenizer_path data/llama  --destination_path data/SP_tokenized --split train --percentage 1.0
python scripts/prepare_slimpajama.py --source_path data/SlimPajama-627B --tokenizer_path data/llama  --destination_path data/SP_tokenized --split validation --percentage 1.0
```
The processed training data (chunk 1 to chunk 4) has in total XX tokens. Making our experiments in the infinite data regime (no training tokens will ever been seen more than once). 

### Pretraining
If your setup comprises two nodes, each with 8 GPUs, you can initiate pretraining with the following commands:

On node 1:
```
lightning run model \
    --devices=8 \
    pretrain/tinyllama.py 
```
