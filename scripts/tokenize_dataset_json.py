import json
import logging
import sys
from pathlib import Path
from typing import Optional, Literal
from transformers import AutoTokenizer
import torch
from torch.utils.data import random_split
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
logger = logging.getLogger(__name__)
sys.path.append(str(wd))

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def prepare(
    json_path: Path,
    destination_path: Path = Path("data/open_assistant"),
    checkpoint_dir: Path = Path("checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T"),
    mask_inputs: bool = False,
    ignore_index: int = -1,
    max_seq_length: Optional[int] = None,
) -> None:
    """Prepare a CSV dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    if max_seq_length is None:
        with open(checkpoint_dir / "lit_config.json", "r") as file:
            config = json.load(file)
            max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)
    print("Loading json...")
    train_set = []
    with open(json_path, 'r') as file:
        for line in file:
            train_set.append(json.loads(line))
    print(f"FT dataset has {len(train_set)} samples")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    print("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]
    torch.save(train_set, destination_path / "train.pt")


def prepare_sample(example, tokenizer, max_length: int, mask_inputs: bool, ignore_index: int) -> dict:

    full_prompt = tokenizer.apply_chat_template(
        example, tokenize=False, add_generation_prompt= False
    )
    full_prompt_ids = tokenizer.encode(full_prompt, return_tensors="pt"  ,add_special_tokens=False)[0][:max_length]
    labels = full_prompt_ids.clone()
    if mask_inputs:
        #TODO
        raise NotImplementedError
    return {
        "input_ids": full_prompt_ids,
        "labels": labels,
    }



if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare, as_positional=False)