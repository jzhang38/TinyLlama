## How to fine-tune TinyLlama?

step 1: Download base model weights from Huggingface and convert them into Lit-GPT format

```bash
python scripts/download.py \
	--repo_id TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T
```

```bash
python scripts/convert_hf_checkpoint.py \
	--checkpoint_dir checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
	--model_name tiny_LLaMA_1b
```

- step 2: Prepare fine-tuning dataset (Now only support OpenAssistant)

```bash
# Download dataset from Huggingface and convert it into csv format
python scripts/convert_hf_dataset_to_csv.py
```
```bash
python scripts/prepare_csv.py \
  --csv_path data/raw_csv/open_assistant.csv \
  --destination_path data/open_assistant \
  --checkpoint_dir checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
  --max_seq_length 512
```

- step 3: Fine-tuning

```bash
python sft/ft.py \
  --data_dir data/open_assistant \
  --model_name tiny_LLaMA_1b \
  --checkpoint_dir checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
  --out_dir out/oasst/ \
```
