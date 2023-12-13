# LLM Scaling Law Made Easy

## Installation
This code base is tested on RTX4090 with CUDA 11.8.
#### Install Pytorch & Xformers
```bash
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
```

#### Install Flash-Attention 2 and other fused operators:
```bash
pip uninstall ninja -y && pip install ninja -U
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install flash-attn --no-build-isolation
cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../.. && rm -rf flash-attention
```
#### Install Remaining Dependencies
```
pip install -r requirements.txt 
```

## Data Preparation
You need at least 1T disk space to download data and reproduce all our experiments.
#### Download Datasets
Download the Slimpajama to your chosen directory. 
```bash
mkdir data && cd data
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
rm SlimPajama-627B/.git
git clone https://huggingface.co/datasets/bigcode/starcoderdata
rm starcoderdata-627B/.git
cd ..
```
Note that to save space, we only use chunk 1 to chunk 4 in the training set for our experiments. This operation does not lose generalizability, because SlimPajama is already shuffled.
#### Tokenize data

Use the provided scripts to tokenize the datasets.
```bash
mkdir data && cd data && mkdir llama && cd llama && wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T/blob/main/tokenizer.model && cd ../..
python scripts/prepare_slimpajama.py --source_path data/SlimPajama-627B --tokenizer_path data/llama  --destination_path data/slim_star_combined --split train --percentage 1.0
python scripts/prepare_slimpajama.py --source_path data/SlimPajama-627B --tokenizer_path data/llama  --destination_path data/slim_star_combined --split validation --percentage 1.0
python scripts/prepare_starcoder.py --source_path data/starcoderdata --tokenizer_path data/llama --destination_path data/slim_star_combined --split train --percentage 1.0
```
The processed training data (chunk 1 to chunk 4) has in total 275B tokens, making our experiments in the infinite data regime (no training tokens will ever been seen more than once). 

## Training
```
lightning run model --devices=8 main.py --training_config training_configs/tinyllama.yaml
```
## Downstream Evaluation
Below command generate HF weight and config.json.
```
python scripts/convert_lit2hf_checkpoint.py --out_dir [checkpoint_save_folder_name] --checkpoint_name [step-XXXXX-ckpt.pth] --model_name  [model_name]
```
The remaining files is exactly the same as Llama-2-7B(https://huggingface.co/meta-llama/Llama-2-7b).
To run eval, clone [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) and run:
```
python main.py \
    --model hf-causal \
    --model_args pretrained=[model_path],dtype="float" \
    --tasks lambada_openai\
    --device cuda:0 --batch_size 16
```
