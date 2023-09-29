python scripts/convert_hf_checkpoint.py --checkpoint_dir  out/TinyLlama-1.1B-intermediate-900B --model_name tiny_LLaMA_1b

python test_weight.py --checkpoint_dir out/TinyLlama-1.1B-intermediate-900B


python pretrain/tinyllama_code.py --devices 8 --train_data_dir data/code_specialist_python_java_javascript_8192



python scripts/prepare_starcoder.py --source_path data/starcoderdata/ --tokenizer_path data/llama --destination_path data/code_specialist_python_java_javascript_8192 --split train --percentage 1.0 --filenames_subset ["python","java","javascript"] --chunk_size 4194816