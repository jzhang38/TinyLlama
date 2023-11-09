import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
from pathlib import Path
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Tokenizer
import random
import pandas as pd


def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    split: str="train",
    filenames_subset: List[str] = None,
    process_id: int = 0
) -> None:

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset 
    
    if not filenames:
        raise RuntimeError(
            f"No files matching  found at {source_path}. \n"
            "Make sure you download the data..."
        )

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_starcoder_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )
    
    total_contents = []
    print("--------------------------------------------------------------------------------")
    print(f"Process {process_id} start loading {len(filenames)} data chunks into memory")
    for filepath in filenames:
        total_contents += pd.read_parquet(filepath, engine='pyarrow')['content'].tolist()
    random.seed(1024)
    random.shuffle(total_contents)
    print("--------------------------------------------------------------------------------")
    print(f"Process {process_id} start tokenizing")
    for text in total_contents:
        text_ids = tokenizer.encode(text)
        builder.add_array(np.array(text_ids, dtype=builder.dtype))

    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids, see https://github.com/jzhang38/TinyLlama/issues/83 for more details
    # builder.write_reminder()


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/"),
    destination_path: Path = Path("data/red_pajama_sample"),
    chunk_size: int = 4097 * 1024,
    split: str="train",
    percentage: float = 1.0,
    filenames_subset: List[str] = None,
) -> None:
    import time
    assert split == "train" #  starcoder only has train data
    filenames = glob.glob(os.path.join(source_path, "*/*.parquet"), recursive=True)
    random.seed(1024)
    random.shuffle(filenames)
    # only retrain subsets that follow the prefix in filenames_subset
    if filenames_subset:
        filenames = [f for f in filenames if any([prefix== Path(f).parent.name for prefix in filenames_subset])]
    filenames = filenames[:int(len(filenames) * percentage)]
    num_processes = 53
    chunked_filenames = np.array_split(filenames, num_processes)

    processes = []
    start_time = time.time()

    # prepare_full(source_path, tokenizer_path, destination_path, chunk_size, split,filenames, 0)
    for i, subset in enumerate(chunked_filenames):
        p = Process(target=prepare_full, args=(source_path, tokenizer_path, destination_path, chunk_size, split, list(subset), i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)
