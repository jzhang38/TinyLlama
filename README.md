<div align="center">

# TinyLlama-1.1B
English | [中文](README_zh-CN.md)
</div>

The TinyLlama project aims to **pretrain** a **1.1B Llama model on 3 trillion tokens**. With some proper optimization, we can achieve this within a span of "just" 90 days using 16 A100-40G GPUs 🚀🚀. The training has started on 2023-09-01. 

<div align="center">
  <img src=".github/TinyLlama_logo.png" width="300"/>
</div>

We adopted exactly the same architecture and tokenizer as Llama 2. This means TinyLlama can be plugged and played in many open-source projects built upon Llama. Besides, TinyLlama is compact with only 1.1B parameters. This compactness allows it to cater to a multitude of applications demanding a restricted computation and memory footprint.


#### Releases Schedule
We will be rolling out intermediate checkpoints following the below schedule. We also include some baseline models for comparison.

| Date       | HF Checkpoint                                   | Tokens | Step | HellaSwag Acc_norm |
|------------|-------------------------------------------------|--------|------|---------------------|
| Baseline   | [StableLM-Alpha-3B](https://huggingface.co/stabilityai/stablelm-base-alpha-3b)| 800B   | --   |  38.31            |
| Baseline   | [Pythia-1B-intermediate-step-50k-105b](https://huggingface.co/EleutherAI/pythia-1b/tree/step50000)             | 105B   | 50k   |  42.04            |
| Baseline   | [Pythia-1B](https://huggingface.co/EleutherAI/pythia-1b)             | 300B   | 143k   |  47.16            |
| 2023-09-04 | [TinyLlama-1.1B-intermediate-step-50k-105b](https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b) | 105B   | 50k   |  43.50               |
| 2023-09-16 | --                                             | 500B   | --   |  --               |
| 2023-10-01 | --                                             | 1T     | --   |  --               |
| 2023-10-16 | --                                             | 1.5T   | --   |  --               |
| 2023-10-31 | --                                             | 2T     | --   |  --               |
| 2023-11-15 | --                                             | 2.5T   | --   |  --               |
| 2023-12-01 | --                                             | 3T     | --   |  --               |

<!-- | Baseline   | [Pythia-1B-intermediate-52b](https://huggingface.co/EleutherAI/pythia-1b/tree/step25000)             | 52B   | 25k   |  38.81            | -->
<!-- | Baseline   | [Pythia-1.4B-intermediate-52b](https://huggingface.co/EleutherAI/pythia-1.4b/tree/step25000)             | 52B   | 25k   |  42.49            | -->
<!-- | Baseline   | [Pythia-1.4B-intermediate-105b](https://huggingface.co/EleutherAI/pythia-1.4b/tree/step50000)             | 105B   | 50k   |  46.14            | -->
<!-- | 2023-09-04 | [TinyLlama-1.1B-intermediate-52b](https://huggingface.co/PY007/TinyLlama-1.1B-52b)   | 52B    | 25k  |  40.85            |
| 2023-09-04 | [TinyLlama-1.1B-intermediate-84b](https://huggingface.co/PY007/TinyLlama-1.1B-84b)   | 84B    | 40k  |  42.65            |  -->

It can be observed that TinyLlama has so far progressed well 🎉🎉. 

Meanwhile, you can track the live cross entropy loss [here](https://wandb.ai/lance777/lightning_logs/reports/metric-train_loss-23-09-02-15-26-17---Vmlldzo1MjkzNzMw?accessToken=9843chbl7rfi1w03hxttpcnbo9z8t6088pw3ddn4h8teunaq0cy7j8hw9c5i02ve).

## Potential Usecase
Tiny but strong language models are useful for many applications. Here are some potential usecases:
- Assisting speculative decoding of larger models. (See this [tutorial](https://twitter.com/karpathy/status/1697318534555336961) by Andrej Karpathy)
- Deployment on edge devices with restricted memory and computational capacities, for functionalities like real-time machine translation without an internet connection (the 4bit-quantized TinyLlama-1.1B's weight only takes up 550MB RAM).
- Enabling real-time dialogue generation in video games.

Moreover, our code can be a valuable **reference for enthusiasts keen on pretraining language models under 5 billion parameters** without diving too early into [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

## Training Details
Below are some details of our training setup:

| Setting                         | Description                                                    |
|---------------------------------|----------------------------------------------------------------|
| Parameters                      | 1.1B                                                           |
| Attention Variant               | Grouped Query Attention                                        |
| Model Size                      | Layers: 22, Heads: 32, Query Groups: 4, Embedding Size: 2048, Intermediate Size (Swiglu): 5632|
| Sequence Length                 | 2048                                                           |
| Batch Size                      | 2 million tokens (2048 * 1024)                                             |
| Learning Rate                   | 4e-4                                                           |
| Learning Rate Schedule          | Cosine with 2000 warmup steps                                  |
| Training Data                   | [Slimpajama](https://huggingface.co/datasets/cerebras/slimpajama-627b) & [Starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) |
| Data Preprocessing              | Excluded GitHub subset of Slimpajama; Sampled all code from Starcoderdata |
| Combined Dataset Size           | 1 trillion tokens                                              |
| Total Tokens During Training    | 3 trillion (3 epochs/1430k steps)                                          |
| Natural Language to Code Ratio  | 7:3                                                            |
| Hardware                        | 16 A100-40G GPUs                                               |






## Blazingly Fast
Our codebase supports the following features:
- multi-gpu and multi-node distributed training with FSDP.
- flash attention 2.
- fused layernorm.
- fused swiglu.
- fused cross entropy loss .
- fused rotary positional embedding.

Thanks to those optimizations, we achieve a throughput of **24k** tokens per second per A100-40G GPU, which translates to **56% model flops utilization** without activation checkpointing (We expect the MFU to be even higher on A100-80G). It means you can train a chinchilla-optimal TinyLlama (1.1B param, 22B tokens) in **32 hours with 8 A100**. Those optimizations also greatly reduce the memory footprint, allowing us to stuff our 1.1B model into 40GB GPU RAM and train with a per-gpu batch size of 16k tokens. **You can also pretrain TinyLlama on 3090/4090 GPUs with a smaller per-gpu batch size**.
Below is a comparison of the training speed of our codebase with that of Pythia and MPT.


| Model                             | A100 GPU hours taken on 300B tokens| 
|-----------------------------------|------------------------------------|
|TinyLlama-1.1B                     | 3456                               |    
|[Pythia-1.0B](https://huggingface.co/EleutherAI/pythia-1b)                        | 4830                               |
|[MPT-1.3B](https://huggingface.co/mosaicml/mpt-1b-redpajama-200b)                           | 7920                               |  

<small> The Pythia number comes from their [paper](https://arxiv.org/abs/2304.01373). The MPT number comes from [here](https://huggingface.co/mosaicml/mpt-1b-redpajama-200b), in which they say MPT-1.3B " was trained on 440 A100-40GBs for about half a day" on 200B tokens. </small>

The fact that TinyLlama is a relatively small model with grouped query attention means it is also fast during inference. Below are some throughputs that we measure:

| Framework | Device | Batch Size | Throughput |
|-----------|--------------|-----|-----------|
|[Llama.cpp](https://github.com/ggerganov/llama.cpp) | Mac M2 16GB RAM         |  1|    71.8 tokens/sec      | 
|[vLLM](https://github.com/vllm-project/vllm)       | One A40 GPU  |  |           |


## Getting Started
Please refer to [PRETRAIN.md](PRETRAIN.md) for instructions on how to pretrain TinyLlama.

## TODO
This project is still under active development. We are a really small team. Community feedback and contributions are highly appreciated. Here are some things we plan to work on:
 - [ ] Add scripts for pretraining on other datasets.
 - [ ] Sequence length extrapolation.
 - [ ] Test the throughput on RTX 3090/4090. 
 - [ ] Add fine-tuning scripts.
 - [ ] Properly evaluate the model on downstream tasks.
 - [ ] A demo running on mobile phones. 
 - [ ] Explore retrieval-augmentation.


## Acknowledgements
This repository is built upon [lit-gpt](https://github.com/Lightning-AI/lit-gpt) and [flash-attention](https://github.com/Dao-AILab/flash-attention). Be sure to explore this fantastic open-source project if it's new to you!
```
@online{lit-gpt,
  author    = {Lightning AI},
  title     = {Lit-GPT},
  url       = {https://github.com/Lightning-AI/lit-gpt},
  year      = {2023},
}
@article{dao2023flashattention2,
  title     ={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author    ={Dao, Tri},
  year      ={2023}
}
```

## Citation
This project is currently contributed by [Peiyuan Zhang](https://github.com/jzhang38), [Guangtao Zeng](https://github.com/ChaosCodes), [Tianduo Wang](https://github.com/TianduoWang) and [Wei Lu](https://istd.sutd.edu.sg/people/faculty/lu-wei/). 

If you find our work valuable, please cite:

```
@online{tinyllama,
  author    = {Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu},
  title     = {TinyLlama},
  url       = {https://github.com/jzhang38/TinyLlama},
  year      = {2023},
  month     = {Sep},
}
```

## Frequently Asked Questions

#### 1. Why would pretraining a 1.1B model for so long make sense? Doesn't it contradict the Chinchilla Scaling Law?

<img src=".github/llama2-training.png" alt="The training loss curve of Llama 2" width="500"/>

Above is the training loss curve taken from the Llama 2 paper. Here I quote from that paper: "We observe that after pretraining on 2T Tokens, the models still did not show any sign of saturation". That is why we believe pretraining a 1.1B model for 3T tokens is a reasonable thing to do. Even if the loss curve does not go down eventually, we can still study the phenomenon of saturation and learn something from it.

#### 2. What does "saturation" mean?
<img src=".github/Pythia_saturation.png" alt="Figure 10 of the Pythia paper" width="500"/>

The figure from the Pythia paper displays the LAMBADA accuracy plotted against the total training tokens (300B). The term "saturation" pertains specifically to the 70M and 160M models. Notably, even the 410M model does not saturate with 300B tokens, as it continues to show an increasing trend, similar to the trend of larger models. 
