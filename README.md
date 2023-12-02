# LLM Scaling Law Made Easy

## Installation
This code base is tested on RTX4090 with CUDA 11.8.
#### Install Pytorch
```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
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
Note that to save space, we only use chunk 1 to chunk 4 in the training set for our experiments. This operation does not lose generalizability, because SlimPajama is already shuffled.
#### Tokenize data

Use the provided scripts to tokenize the datasets and divide them into chunks.
```bash
mkdir data && cd data && mkdir llama && cd llama && wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T/blob/main/tokenizer.model && cd ../..
python scripts/prepare_slimpajama.py --source_path data/SlimPajama-627B --tokenizer_path data/llama  --destination_path data/SP_tokenized --split train --percentage 1.0
python scripts/prepare_slimpajama.py --source_path data/SlimPajama-627B --tokenizer_path data/llama  --destination_path data/SP_tokenized --split validation --percentage 1.0
```
The processed training data (chunk 1 to chunk 4) has in total 275B tokens, making our experiments in the infinite data regime (no training tokens will ever been seen more than once). 

## Training
If your setup comprises two nodes, each with 8 GPUs, you can initiate pretraining with the following commands:

On node 1:
```
python pretrain/tinyllama.py --training_config XXX.yaml
```

## Downstream Evaluation
Below command generate HF weight and config.json.
```
python scripts/convert_lit2hf_checkpoint.py --out_dir out/llama_19M_lr_1e-3_bs_0.5M_step_16K --checkpoint_name iter-012800-ckpt.pth --model_name  llama_19M
```
The remaining files is exactly the same as Llama-2-7B(https://huggingface.co/meta-llama/Llama-2-7b).
To run eval, clone [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) and run:
```
python main.py \
    --model hf-causal \
    --model_args pretrained=/root/ScalingLaw/data/test_model,dtype="float" \
    --tasks lambada_openai\
    --device cuda:9 --batch_size 16
```