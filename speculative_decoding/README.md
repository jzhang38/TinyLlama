## Speculative Decoding

### HuggingFace "Assisted Generation"


| Large Model | Native Decoding | Assisted Decoding  |
| ----------- | --------------- | ------------------ |
| guanaco-7b  | 69  seconds   | 38 seconds      |
| guanaco-13b | 84 seconds             | 45 seconds                 | 
| guanaco-33b | 109 seconds             | 62 seconds                 | 

We use PY007/TinyLlama-1.1B-Chat-v0.1 as the assistant model and vary the large model from guanaco-7B to 33B. Experiments are done on a single A40 GPU with code inside instruct_hf_assisted_decoding.py. TinyLlama is loaded in fp16 and the large models are loaded in 8 bit to make guanaco-33b fit in memory and also to keep a consistent setup. The prompt used is "Give me detailed info about Jeo Biden.". max_new_tokens is set to 512. 

You can read this [article](https://huggingface.co/blog/assisted-generation) for more information about HuggingFace's Assisted Generation.

Quote from HF: "due to INT8 quantization and the use of causal masking in assisted generation, the output of greedy decoding may differ in rare occasions."
#### TODO
- [ ] Thouroughly benchmark the average speedup on 52K Alpaca prompts.

### Llama.cpp Speculative Decoding
We are currently having issue with correctly converting the model weight. See [issue 24](https://github.com/jzhang38/TinyLlama/issues/24) for more information. If you have any idea, please let us know.