<div align="center">

# TinyLlama-1.1B
[English](README.md) | 中文
</div>

TinyLlama项目旨在在30万亿tokens上进行预训练，构建一个拥有11亿参数的Llama模型。经过精心优化，我们"仅"需16块A100-40G的GPU，便可在90天内完成这个任务🚀🚀。训练已于2023-09-01开始。


<div align="center">
  <img src=".github/TinyLlama_logo.png" width="300"/>
</div>

我们采用了与Llama 2完全相同的架构和分词器。这意味着TinyLlama可以在许多基于Llama的开源项目中即插即用。此外，TinyLlama只有1.1B的参数，体积小巧，适用于需要限制计算和内存占用的多种应用。


#### 发布时间表

我们会根据以下计划逐步发布中间checkpoint。我们也列了一些基线模型进行比较。



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

从上面可以看出，TinyLlama目前的进展非常好🎉🎉。


你也可以在[这里](https://wandb.ai/lance777/lightning_logs/reports/metric-train_loss-23-09-02-15-26-17---Vmlldzo1MjkzNzMw?accessToken=9843chbl7rfi1w03hxttpcnbo9z8t6088pw3ddn4h8teunaq0cy7j8hw9c5i02ve)实时跟踪TinyLlama的训练损失。

## 潜在场景
小型但强大的语言模型对许多应用都很有用。以下是一些潜在的场景：
- 帮助对大型模型进行speculative decoding。
- 在边缘装置上运行，比如离线的实时机器翻译 (TinyLlama的4比特量化版本的模型权重只需要550MB的内存)。
- 在游戏中实现实时对话生成(因为还得给游戏本身留显存所以模型要小)。

此外，我们的代码可以给初学者做一个**入门预训练的简洁参考**。如果你要训练50亿以下参数的语言模型, 你其实不需要Megatron-LM。

## 训练细节
以下是我们训练设置的一些细节：

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
| Total Tokens During Training    | 3 trillion (3 epochs/143k steps)                                          |
| Natural Language to Code Ratio  | 7:3                                                            |
| Hardware                        | 16 A100-40G GPUs                                               |






## 速度极快
我们的代码库支持以下特性：
- multi-gpu and multi-node distributed training with FSDP.
- flash attention 2.
- fused layernorm.
- fused swiglu.
- fused cross entropy loss .
- fused rotary positional embedding.

有了这些优化, 我们可以达到**24k tokens/秒/A100**的训练速度，也就是56%的MFU（在A100-80G上的MFU会更高）。这个速度可以让你可以在**8个A100上用32小时训练一个chinchilla-optimial的模型**(11亿参数，220亿token)。这些优化也大大减少了显存占用, 我们可以把11亿参数的模型塞入40GB的GPU里面还能同时维持16k tokens的per-gpu batch size。只需要把batch size改小一点， 你就可以在**RTX 3090/4090**上面训练TinyLlama。
下面是我们的代码库与Pythia和MPT的训练速度的比较。


| Model                             | A100 GPU hours taken on 300B tokens| 
|-----------------------------------|------------------------------------|
|TinyLlama-1.1B                     | 3456                               |    
|[Pythia-1.0B](https://huggingface.co/EleutherAI/pythia-1b)                        | 4830                               |
|[MPT-1.3B](https://huggingface.co/mosaicml/mpt-1b-redpajama-200b)                           | 7920                               |  

<small> Pythia的数字来自他们的论文。MPT的数字来自[这里](https://huggingface.co/mosaicml/mpt-1b-redpajama-200b)，作者说MPT-1.3B"was trained on 440 A100-40GBs for about half a day" on 200B tokens。</small>

TinyLlama是一个相对较小的模型, 同时我们用了GQA, 这意味着它在推理期间也很快。以下是我们测量的一些推理速度：

| Framework | Device | Batch Size | Throughput (tokens/sec) |
|-----------|--------------|-----|-----------|
|[Llama.cpp](https://github.com/ggerganov/llama.cpp) | Mac M2 16GB RAM         |  batch_size=1; 4-bit inference|    71.8     | 
|[vLLM](https://github.com/vllm-project/vllm)       | A40 GPU  | batch_size=100, n=10 |   7094.5         |


## 开始训练
请参考[PRETRAIN.md](PRETRAIN.md)。

## TODO
该项目仍在积极开发中。我们团队很小，非常欢迎社区的反馈和贡献。以下是我们计划进行的一些工作：
 - [ ] Add scripts for pretraining on other datasets.
 - [ ] Sequence length extrapolation.
 - [ ] Test the throughput on RTX 3090/4090. 
 - [ ] Add fine-tuning scripts.
 - [ ] Properly evaluate the model on downstream tasks.
 - [ ] A demo running on mobile phones. 
 - [ ] Explore retrieval-augmentation.


## Acknowledgements
这个仓库基于出色的开源项目[lit-gpt](https://github.com/Lightning-AI/lit-gpt)和[flash-attention](https://github.com/Dao-AILab/flash-attention)构建. 
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
此项目目前由[Peiyuan Zhang](https://github.com/jzhang38)，[Guangtao Zeng](https://github.com/ChaosCodes)，[Tianduo Wang](https://github.com/TianduoWang)和[Wei Lu](https://istd.sutd.edu.sg/people/faculty/lu-wei/)贡献。 

如果您觉得我们的工作有价值， 可以引用:

```
@online{tinyllama,
  author    = {Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu},
  title     = {TinyLlama},
  url       = {https://github.com/jzhang38/TinyLlama},
  year      = {2023},
  month     = {Sep},
}
```
