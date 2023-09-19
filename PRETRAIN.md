## Pretrain TinyLlama

### Installation
We expect you have CUDA 11.8 installed.
#### Install Pytorch Nightly.
```bash
pip install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'
```
#### Build XFormers from Source
Note: as of 2023/09/02, xformers does not provide pre-built binaries for torch 2.1. You have to build it from source.
```bash
pip uninstall ninja -y && pip install ninja -U
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```


#### Install Flash-Attention 2 and other fused operators:
```bash
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
pip install -r requirements.txt tokenizers sentencepiece
```
to install other dependencies.
It may take >= 5 minutes to build xformers/flash-attention. Do not worry if the process seemly stagnant or the terminal print out many warnings.

Then you are ready to go 🎉!

### Data Preparation

#### Download Datasets
Download the Slimpajama and Starcoderdata datasets to your chosen directory.
```bash
cd /path/to/dataset
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
git clone https://huggingface.co/datasets/bigcode/starcoderdata
```
The SlimPajama dataset eats 893GB diskspace and the starcoderdata takes 290GB.

#### Tokenize data
Use the provided scripts to tokenize the datasets and divide them into chunks.
```bash
python scripts/prepare_starcoder.py --source_path /path/to/starcoderdata/ --tokenizer_path data/llama --destination_path data/slim_star_combined --split train --percentage 1.0
python scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama --tokenizer_path data/llama  --destination_path data/slim_star_combined --split validation --percentage 1.0
python scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama --tokenizer_path data/llama  --destination_path data/slim_star_combined --split train --percentage 1.0
```
The processed data will take 1.8T storage.

### Pretraining
If your setup comprises two nodes, each with 8 GPUs, you can initiate pretraining with the following commands:

On node 1:
```
lightning run model \
    --node-rank=0  \
    --main-address=172.16.101.5 \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=2 \
    pretrain/tinyllama.py --devices 8 --train_data_dir data/slim_star  --val_data_dir data/slim_star
```
On node 2:
```
lightning run model \
    --node-rank=1  \
    --main-address=172.16.101.5 \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=2 \
    pretrain/tinyllama.py --devices 8 --train_data_dir data/slim_star   --val_data_dir data/slim_star
```
You can follow [these instructions](https://lightning.ai/docs/fabric/stable/guide/multi_node/slurm.html) if you have a slurm cluster.

