## How to fine-tune TinyLlama?

- step 1: Download base model weights from Huggingface and convert them into Lit-GPT format
bash```
python scripts/download.py --repo_id TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T
```
bash```
python scripts/convert_hf_checkpoint.py 
	--checkpoint_dir checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T 
	--model_name tiny_LLaMA_1b
```

- step 2: Prepare fine-tuning dataset (take Dolly-v2 dataset as example)
bash```
python scripts/prepare_dolly.py \
  --destination_path data/dolly \
  --checkpoint_dir checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T
```

- step 3: Fine-tuning
bash```
python sft/ft.py \
  --data_dir data/dolly \
  --model_name tiny_LLaMA_1b \
  --checkpoint_dir checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
  --out_dir out/dolly/
```