import json
import glob
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import multiprocessing
import logging
import zstandard as zstd
import configparser


SLIMPAJAMA_SETS = {
    "train": "train/chunk*/*",
    "validation": "validation/chunk*/*",
    "test": "test/chunk*/*",
}

def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    split: str = "train",
    filenames_subset: List[str] = None,
    process_id: int = 0
) -> None:
    try:
        
        destination_path = destination_path.resolve()
        destination_path.mkdir(parents=True, exist_ok=True)

        
        logger = logging.getLogger(f"process_{process_id}")
        logger.setLevel(logging.INFO)
        log_handler = logging.FileHandler(f"process_{process_id}.log")
        log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        log_handler.setFormatter(log_formatter)
        logger.addHandler(log_handler)

        logger.info(f"Starting data preparation for {split} in process {process_id}")

        tokenizer = Tokenizer(tokenizer_path)

       
        if not filenames_subset:
            raise RuntimeError(
                f"No files matching {SLIMPAJAMA_SETS[split]} found at {source_path}. \n"
                "Make sure you download the data..."
            )

        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=f"{split}_slimpajama_{process_id}",
            chunk_size=chunk_size,
            sep_token=tokenizer.bos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )

        for filepath in filenames_subset:
            logger.info(f"Processing {filepath}")
            with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                for row in tqdm(f):
                    text = json.loads(row)["text"]
                    if json.loads(row)["meta"]["redpajama_set_name"] == "RedPajamaGithub":
                        continue
                    text_ids = tokenizer.encode(text)
                    builder.add_array(np.array(text_ids, dtype=builder.dtype))

        builder.write_reminder()
        logger.info(f"Data preparation completed for {split} in process {process_id}")

    except Exception as e:
        logger.error(f"Error processing {split} data in process {process_id}: {e}")

def prepare(
    config_path: Path = Path("config.ini")
) -> None:
    try:
        
        logging.basicConfig(
            filename="data_preparation.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        config = configparser.ConfigParser()
        config.read(config_path)
        
        source_path = Path(config.get("Paths", "source_path"))
        tokenizer_path = Path(config.get("Paths", "tokenizer_path"))
        destination_path = Path(config.get("Paths", "destination_path"))
        chunk_size = int(config.get("Options", "chunk_size"))
        split = config.get("Options", "split")
        percentage = float(config.get("Options", "percentage"))

        filenames = glob.glob(os.path.join(source_path, SLIMPAJAMA_SETS[split]), recursive=True)
        filenames = filenames[:int(len(filenames) * percentage)]

        num_processes = multiprocessing.cpu_count() 
        chunked_filenames = np.array_split(filenames, num_processes)

        processes = []
        start_time = time.time()

        for i, subset in enumerate(chunked_filenames):
            p = multiprocessing.Process(target=prepare_full, args=(source_path, tokenizer_path, destination_path, chunk_size, split, list(subset), i))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Data preparation for {split} completed. Time taken: {elapsed_time:.2f} seconds")

    except Exception as e:
        logging.error(f"Error in preparing data: {e}")

if __name__ == "__main__":
    try:
        prepare()
    except Exception as e:
        logging.error(f"Main process error: {e}")
