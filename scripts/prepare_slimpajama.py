import json
import glob
import os
import sys
import time
import numpy as np

from tqdm import tqdm
from multiprocessing import Process, cpu_count
from pathlib import Path
from typing import List

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Tokenizer

# Filename for SlimPajama
slimpajama_sets = {
    "train": "train/chunk*/*",
    "validation": "validation/chunk*/*",
    "test": "test/chunk*/*",
}


def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    split: str="train",
    filenames_subset: List[str] = None,
    process_id: int = 0,
    retry_attempt: int = 0
) -> None:
    import zstandard as zstd

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset 
    
    if not filenames:
        raise RuntimeError(
            f"No files matching {slimpajama_sets[split]} found at {source_path}. \n"
            "Make sure you download the data..."
        )

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_slimpajama_{process_id}_{retry_attempt}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    for filepath in filenames:
        print(f"Processing {filepath}")
        with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
            for row in tqdm(f):
                text = json.loads(row)["text"]
                if json.loads(row)["meta"]["redpajama_set_name"] == "RedPajamaGithub":
                    continue # we don't want to include the github data
                text_ids = tokenizer.encode(text)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))

    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids, see https://github.com/jzhang38/TinyLlama/issues/83 for more details
    # builder.write_reminder()

def is_process_running(pid):
    """
    Checks if a process with the given PID is still running on Linux.

    Args:
        pid: The process ID (integer) to check.

    Returns:
        True if the process is still running, False otherwise.
    """
    try:
        # Attempt to send signal 0 (check if process exists)
        os.kill(pid, 0)
        return True
    except OSError:
        # ProcessDoesNotExist exception will be raised if process is not found
        return False

def download_files(
    filenames,
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    destination_path: Path = Path("data/red_pajama_sample"),
    chunk_size: int = 2049 * 1024,
    split: str="train",
    retry_attempt: int = 0
):
    num_processes = cpu_count() 
    chunked_filenames = np.array_split(filenames, num_processes)

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        p = Process(target=prepare_full, args=(source_path, tokenizer_path, destination_path, chunk_size, split, list(subset), i, retry_attempt))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    destination_path: Path = Path("data/red_pajama_sample"),
    chunk_size: int = 2049 * 1024,
    split: str="train",
    percentage: float = 1.0,
    data_download_pid: int = None
) -> None:

    filenames = glob.glob(os.path.join(source_path, slimpajama_sets[split]), recursive=True)
    filenames = filenames[:int(len(filenames) * percentage)]

    if data_download_pid is None or not is_process_running(data_download_pid):
        print("Start regular processing")
        download_files(
            filenames,
            source_path,
            tokenizer_path,
            destination_path,
            chunk_size,
            split
        )
    else:
        retry_attempt = 0
        last_attempt = False
        while is_process_running(data_download_pid) or last_attempt:
            print(f"Processing while downloading data, attempt {retry_attempt} last attempt {last_attempt}")
            # get a set of current file names
            new_filenames = []
            if retry_attempt > 0:
                cur_filenames = glob.glob(os.path.join(source_path, slimpajama_sets[split]), recursive=True)
                cur_filenames = cur_filenames[:int(len(cur_filenames) * percentage)]
                for cur_f in cur_filenames:
                    if cur_f not in filenames:
                        new_filenames.append(cur_f)
            else:
                # get existing files for first attempt
                new_filenames = filenames
            
            if len(new_filenames) > 0:
                download_files(
                    new_filenames,
                    source_path,
                    tokenizer_path,
                    destination_path,
                    chunk_size,
                    split,
                    retry_attempt
                )
            else:
                print("No new files found")

            retry_attempt += 1
            time.sleep(60) # seconds

            # In case download is finished while process the data, try it one more time
            if last_attempt:
                # reset to prevent infinite loop
                last_attempt = False
            else:
                last_attempt = not is_process_running(data_download_pid)

            # add all new files to downloaded list
            filenames += new_filenames

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)