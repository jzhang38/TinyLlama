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
We have continue-pretrained a code tinyllama from the 500B checkpoint with another 7B Python data [here](https://huggingface.co/PY007/TinyLlama-1.1B-python-v0.1).
The code for continue-pretraining can be found in pretrain/tinyllama_code.py

```
./speculative \
-m models/CodeLlama-7b-hf/ggml-model-f16.gguf \
-md models/TinyLlama-1.1B-500B-python/ggml-model-q4_0.gguf \
-p "# Quick-sort implementation in Python and sample usage:" \
-e -ngl 1 -t 4 -n 256 -s 20 --temp 0 --draft 8
```
This gives:

```
encoded   12 tokens in    0.247 seconds, speed:   48.638 t/s
decoded  265 tokens in    7.909 seconds, speed:   33.507 t/s

n_draft   = 16
n_predict = 265
n_drafted = 317
n_accept  = 195
accept    = 61.514%

draft:

llama_print_timings:        load time =    53.14 ms
llama_print_timings:      sample time =   652.62 ms /     1 runs   (  652.62 ms per token,     1.53 tokens per second)
llama_print_timings: prompt eval time =    73.81 ms /    12 tokens (    6.15 ms per token,   162.58 tokens per second)
llama_print_timings:        eval time =  2247.77 ms /   378 runs   (    5.95 ms per token,   168.17 tokens per second)
llama_print_timings:       total time =  8154.92 ms

target:

llama_print_timings:        load time =   534.47 ms
llama_print_timings:      sample time =   208.12 ms /   265 runs   (    0.79 ms per token,  1273.32 tokens per second)
llama_print_timings: prompt eval time =  4210.38 ms /   382 tokens (   11.02 ms per token,    90.73 tokens per second)
llama_print_timings:        eval time =   682.80 ms /    16 runs   (   42.68 ms per token,    23.43 tokens per second)
llama_print_timings:       total time =  8214.11 ms
ggml_metal_free: deallocating
ggml_metal_free: deallocating
```

Even though the model is continue-pretrained exclusively on Python, it retains its ability in other languages, such as C:
```
./speculative \
-m models/CodeLlama-7b-hf/ggml-model-f16.gguf \
-md models/TinyLlama-1.1B-500B-python/ggml-model-q4_0.gguf \
-p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" \
-e -ngl 1 -t 4 -n 256 -s 20 --temp 0 --draft 8
```

This gives:

```
encoded   25 tokens in    0.278 seconds, speed:   89.900 t/s
decoded  258 tokens in    6.432 seconds, speed:   40.112 t/s

n_draft   = 28
n_predict = 258
n_drafted = 278
n_accept  = 200
accept    = 71.942%

draft:

llama_print_timings:        load time =   932.54 ms
llama_print_timings:      sample time =   583.50 ms /     1 runs   (  583.50 ms per token,     1.71 tokens per second)
llama_print_timings: prompt eval time =    81.50 ms /    25 tokens (    3.26 ms per token,   306.73 tokens per second)
llama_print_timings:        eval time =  1834.67 ms /   329 runs   (    5.58 ms per token,   179.32 tokens per second)
llama_print_timings:       total time =  6710.30 ms

target:

llama_print_timings:        load time = 18568.44 ms
llama_print_timings:      sample time =   208.78 ms /   258 runs   (    0.81 ms per token,  1235.75 tokens per second)
llama_print_timings: prompt eval time =  3164.84 ms /   342 tokens (    9.25 ms per token,   108.06 tokens per second)
llama_print_timings:        eval time =   775.43 ms /    18 runs   (   43.08 ms per token,    23.21 tokens per second)
llama_print_timings:       total time =  7650.67 ms
ggml_metal_free: deallocating
ggml_metal_free: deallocating
```


I have not tried 13B CodeLlama as the large model yet because my Mac memory is not enough :).