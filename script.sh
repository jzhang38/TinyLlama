python scripts/convert_hf_checkpoint.py --checkpoint_dir  out/TinyLlama-1.1B-intermediate-step-240k-503b --model_name tiny_LLaMA_1b

python test_weight.py --checkpoint_dir out/TinyLlama-1.1B-intermediate-step-240k-503b 


python pretrain/shift.py --devices 6 --train_data_dir data/slim_star_combined 