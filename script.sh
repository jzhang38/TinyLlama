python scripts/convert_hf_checkpoint.py --checkpoint_dir  out/TinyLlama-1.1B-intermediate-step-240k-503b --model_name tiny_LLaMA_1b

python test_weight.py --checkpoint_dir out/TinyLlama-1.1B-intermediate-step-240k-503b 


python pretrain/tinyllama_code.py --devices 1 --train_data_dir data/python_only



python scripts/prepare_starcoder.py --source_path data/starcoder/starcoderdata --tokenizer_path data/llama --destination_path python_only --split train --percentage 1 --filenames_subset [python]


python scripts/convert_lit_checkpoint.py --checkpoint_name iter-080000-ckpt.pth --model_name tiny_LLaMA_1b --out_dir out/tiny_LLaMA_1b
