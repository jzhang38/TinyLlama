# This creates a bin saved checkpoint from out_dir/checkpoint_name and save it in out_dir. It also saves a config.json for hg model.
# python scripts/convert_lit_checkpoint.py --out_dir out/tinyllama_500M --checkpoint_name iter-010000-ckpt.pth --model_name tiny_LLaMA_500M --model_only False
# This takes a bin saved checkpoint from out_dir/checkpoint_name and convert to another pth file again?
# python scripts/convert_hf_checkpoint.py --checkpoint_dir  out/tinyllama_500M --model_name tiny_LLaMA_500M

# This will generate iter-160000-ckpt.bin
# OUT_DIR=out/tinyllama_500M
# CHECKPOINT=iter-205000-ckpt
# MODEL_NAME=tiny_LLaMA_500M
OUT_DIR=out/tinyllama_120M
CHECKPOINT=iter-035000-ckpt
MODEL_NAME=tiny_LLaMA_120M
INFERENCE_DIR=out/pretrained
python scripts/convert_lit_checkpoint.py --out_dir $OUT_DIR --checkpoint_name $CHECKPOINT.pth --model_name $MODEL_NAME --model_only False
mv $OUT_DIR/$CHECKPOINT.bin $INFERENCE_DIR/pytorch_model.bin
mv $OUT_DIR/config.json $INFERENCE_DIR/

python pretrain/simple_inference.py

# Not needed
# python scripts/convert_hf_checkpoint.py --checkpoint_dir  out/tinyllama_500M --model_name tiny_LLaMA_500M