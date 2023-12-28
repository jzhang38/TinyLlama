## Evaluate TinyLlama

### GPT4All Benchmarks

We evaluate TinyLlama's commonsense reasoning ability following the [GPT4All](https://gpt4all.io/index.html) evaluation suite. We include Pythia as our baseline. We report the acc_norm by default. 

Base models:

| Model                                     | Pretrain Tokens | HellaSwag | Obqa | WinoGrande | ARC_c | ARC_e | boolq | piqa | avg |
|-------------------------------------------|-----------------|-----------|------|------------|-------|-------|-------|------|-----|
| Pythia-1.0B                               |        300B     | 47.16     | 31.40| 53.43      | 27.05 | 48.99 | 60.83 | 69.21 | 48.30 |
| TinyLlama-1.1B-intermediate-step-50K-104b |        103B     | 43.50     | 29.80| 53.28      | 24.32 | 44.91 | 59.66 | 67.30 | 46.11|
| TinyLlama-1.1B-intermediate-step-240k-503b|        503B     | 49.56     |31.40 |55.80       |26.54  |48.32  |56.91  |69.42  | 48.28 |
| TinyLlama-1.1B-intermediate-step-480k-1007B |     1007B     | 52.54     | 33.40 | 55.96      | 27.82 | 52.36 | 59.54 | 69.91 | 50.22 |
| TinyLlama-1.1B-intermediate-step-715k-1.5T |     1.5T     | 53.68     | 35.20 | 58.33      | 29.18 | 51.89 | 59.08 | 71.65 | 51.29 |
| TinyLlama-1.1B-intermediate-step-955k-2T |     2T     | 54.63     | 33.40 | 56.83      | 28.07 | 54.67 | 63.21 | 70.67 | 51.64 |
| TinyLlama-1.1B-intermediate-step-1195k-2.5T  |     2.5T     | 58.96     | 34.40 | 58.72      | 31.91 | 56.78 | 63.21 | 73.07 | 53.86|
| TinyLlama-1.1B-intermediate-step-1431k-3T  |     3T     | 59.20     | 36.00 | 59.12      | 30.12 | 55.25 | 57.83 | 73.29 | 52.99|


Chat models:
| Model                                     | Pretrain Tokens | HellaSwag | Obqa | WinoGrande | ARC_c | ARC_e | boolq | piqa | avg |
|-------------------------------------------|-----------------|-----------|------|------------|-------|-------|-------|------|-----|
| [TinyLlama-1.1B-Chat-v0.1](https://huggingface.co/PY007/TinyLlama-1.1B-Chat-v0.1)                 |   503B     | 53.81     |32.20 | 55.01  | 28.67 |49.62  | 58.04 | 69.64 | 49.57 |
| [TinyLlama-1.1B-Chat-v0.2](https://huggingface.co/PY007/TinyLlama-1.1B-Chat-v0.2)                 |   503B     | 53.63     |32.80 | 54.85  | 28.75 |49.16  | 55.72 | 69.48 | 49.20 |
| [TinyLlama-1.1B-Chat-v0.3](https://huggingface.co/PY007/TinyLlama-1.1B-Chat-v0.3)                 |   1T       | 56.81     |34.20 | 55.80  | 30.03 |53.20  | 59.57 | 69.91 | 51.36 |
| [TinyLlama-1.1B-Chat-v0.4](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.4)             |   1.5T     | 58.59     |35.40 | 58.80  | 30.80 |54.04  | 57.31 | 71.16 | 52.30 |


We observed huge improvements once we finetuned the model. We attribute this phenomenon to: 1. the base model has not undergone lr cool-down and FT helps to cool down the lr. 2. the SFT stage better elicits the model's internal knowledge.

You can obtain the above scores by running [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness):
```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=PY007/TinyLlama-1.1B-Chat-v0.1,dtype="float" \
    --tasks hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa\
    --device cuda:0 --batch_size 32
```



### Instruct-Eval Benchmarks
We evaluate TinyLlama's ability in problem-solving on the [Instruct-Eval](https://github.com/declare-lab/instruct-eval) evaluation suite. 


| Model                                             | MMLU  | BBH   | HumanEval | DROP  |
| ------------------------------------------------- | ----- | ----- | --------- | ----- |
| Pythia-1.0B                                       | 25.70 | 28.19 | 1.83      | 4.25  |
| TinyLlama-1.1B-intermediate-step-50K-104b         | 26.45 | 28.82 | 5.49      | 11.42 |
| TinyLlama-1.1B-intermediate-step-240k-503b        | 26.16 | 28.83 | 4.88      | 12.43 |
| TinyLlama-1.1B-intermediate-step-480K-1T          | 24.65 | 29.21 | 6.1       | 13.03 |
| TinyLlama-1.1B-intermediate-step-715k-1.5T        | 24.85 | 28.2  | 7.93      | 14.43 |
| TinyLlama-1.1B-intermediate-step-955k-2T          | 25.97 | 29.07 | 6.71      | 13.14 |
| TinyLlama-1.1B-intermediate-step-1195k-token-2.5T | 25.92 | 29.32 | 9.15      | 15.45 |

You can obtain above scores by running [instruct-eval](https://github.com/declare-lab/instruct-eval):
```bash
CUDA_VISIBLE_DEVICES=0 python main.py mmlu --model_name llama --model_path PY007/TinyLlama-1.1B-intermediate-step-480K-1T
CUDA_VISIBLE_DEVICES=1 python main.py bbh --model_name llama --model_path PY007/TinyLlama-1.1B-intermediate-step-480K-1T
CUDA_VISIBLE_DEVICES=2 python main.py drop --model_name llama --model_path PY007/TinyLlama-1.1B-intermediate-step-480K-1T
CUDA_VISIBLE_DEVICES=3 python main.py humaneval  --model_name llama  --n_sample 1 --model_path PY007/TinyLlama-1.1B-intermediate-step-480K-1T
