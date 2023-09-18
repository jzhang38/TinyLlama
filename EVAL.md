## Evaluate TinyLlama

### GPT4All Benchmarks

We evaluate TinyLlama's commonsense reasoning ability fowlling the GPT4All[https://gpt4all.io/index.html] evaluation suite. We include Pythia as our baselines. We report the acc_norm by default. 

| Model                                     | Pretrain Tokens | HellaSwag | Obqa | WinoGrande | ARC_c | ARC_e | boolq | piqa | avg |
|-------------------------------------------|-----------------|-----------|------|------------|-------|-------|-------|------|-----|
| Pythia-1.0B                               |        300B     | 47.16     | 31.40| 53.43      | 27.05 | 48.99 | 60.83 | 69.21 | 48.30 |
| TinyLlama-1.1B-intermediate-step-50K-104b |        103B     | 43.50     | 29.80| 53.28      | 24.32 | 44.91 | 59.66 | 67.30 | 46.11|
| TinyLlama-1.1B-intermediate-step-240k-503b|        503B     | 49.56     |31.40 |55.80       |26.54  |48.32  |56.91  |69.42  | 48.28 |
| TinyLlama-1.1B-Chat-v0.1                  |        503B     | 53.81     |32.20 |55.01       |28.67  |49.62  |58.04  |69.64  | 49.57 |
| TinyLlama-1.1B-Chat-v0.2                  |        503B     | 53.63     |32.80 | 54.85      |28.75  |49.16  | 55.72 |69.48  | 49.20 |

We observed huge improvements once we finetuned the model. We attribute this phenomenon to: 1. the base model has not undergone lr cool-down and FT helps to cool down the lr. 2. the SFT stage better elicit the model's internal knowledge.

You can obtain above scores by running [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness):
```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=PY007/TinyLlama-1.1B-Chat-v0.1,dtype="float" \
    --tasks hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa\
    --device cuda:0 --batch_size 32
```



### Instruct-Eval Benchmarks
We evaluate TinyLlama's ability in problem-solving on the Instruct-Eval[https://github.com/declare-lab/instruct-eval] evaluation suite. 


| Model                                     | MMLU | BBH  | HumanEval | DROP |
|-------------------------------------------|------|------|-----------|------|
| Pythia-1.0B                               | 25.70| 28.19| 1.83      | 4.25 |
| TinyLlama-1.1B-intermediate-step-50K-104b | 26.45|28.82 |5.49       |11.42 |
| TinyLlama-1.1B-intermediate-step-240k-503b|26.16 |  28.83   |4.88       | 12.43|
| TinyLlama-1.1B-Chat-v0.1                  |26.73 |  28.79   | 3.05     |  11.92 |


You can obtain above scores by running [instruct-eval](https://github.com/declare-lab/instruct-eval):
```bash
CUDA_VISIBLE_DEVICES=0 python main.py mmlu --model_name llama --model_path PY007/TinyLlama-1.1B-Chat-v0.1
CUDA_VISIBLE_DEVICES=1 python main.py bbh --model_name llama --model_path PY007/TinyLlama-1.1B-Chat-v0.1
CUDA_VISIBLE_DEVICES=2 python main.py drop --model_name llama --model_path PY007/TinyLlama-1.1B-Chat-v0.1
CUDA_VISIBLE_DEVICES=3 python main.py humaneval  --model_name llama  --n_sample 1 --model_path PY007/TinyLlama-1.1B-Chat-v0.1
