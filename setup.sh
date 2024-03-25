pip install torch==2.1.0 xformers --index-url https://download.pytorch.org/whl/cu121
pip install packaging
pip uninstall ninja -y && pip install ninja -U
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
python setup.py install
cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../..
cd ../..
pip install -r requirements.txt tokenizers sentencepiece

# export PYTHON_PATH="/home/ken/workspace/TinyLlama:$PYTHON_PATH"
# python scripts/prepare_redpajama.py --source_path /home/ken/.cache/huggingface/datasets/togethercomputer___red_pajama-data-1_t-sample/plain_text/1.0.0/6ea3bc8ec2e84ec6d2df1930942e9028ace8c5b9d9143823cf911c50bbd92039 --checkpoint_dir llama-tokenizer  --destination_path data/red_pajama_1_t_sample
# python scripts/prepare_starcoder.py --source_path /home/ken/workspace/data/tinyllama/starcoderdata --tokenizer_path llama-tokenizer --destination_path data/slim_star_combined --split train --percentage 1.0
