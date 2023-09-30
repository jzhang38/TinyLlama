import os
import json
from pathlib import Path
import argparse
import shutil
import tarfile

from download_tools import maybe_download_file

# random 64 examples used with Atlas
nq_64shot = [
    27144,
    14489,
    49702,
    38094,
    6988,
    60660,
    65643,
    48249,
    48085,
    52629,
    48431,
    7262,
    34659,
    24332,
    44839,
    17721,
    50819,
    62279,
    37021,
    77405,
    52556,
    23802,
    40974,
    64678,
    69673,
    77277,
    18419,
    25635,
    1513,
    11930,
    5542,
    13453,
    52754,
    65663,
    67400,
    42409,
    74541,
    33159,
    65445,
    28572,
    74069,
    7162,
    19204,
    63509,
    12244,
    48532,
    72778,
    37507,
    70300,
    29927,
    18186,
    27579,
    58411,
    63559,
    4347,
    59383,
    57392,
    42014,
    77920,
    45592,
    32321,
    3422,
    61041,
    34051,
]

# random 64 examples used with Atlas
triviaqa_64shot = [
    75927,
    38807,
    452,
    68095,
    44621,
    34592,
    36091,
    65286,
    56484,
    48197,
    34692,
    28011,
    16670,
    62641,
    37865,
    6658,
    45724,
    37527,
    17740,
    31133,
    8010,
    48573,
    53670,
    15514,
    25996,
    54404,
    10739,
    55105,
    66122,
    73324,
    41202,
    71253,
    41258,
    51344,
    60092,
    50455,
    65078,
    36169,
    33408,
    55106,
    40526,
    65582,
    66337,
    39766,
    77174,
    17289,
    7367,
    50930,
    21151,
    21809,
    52804,
    26110,
    54414,
    73358,
    11459,
    66019,
    41084,
    13349,
    39059,
    6626,
    25540,
    15110,
    53320,
    61313,
]


def convert_triviaqa(ex):
    target = ex["Answer"]["Value"]
    if target.isupper():
        target = target.title()
    return {
        "question": ex["Question"],
        "answers": ex["Answer"]["Aliases"],
        "target": target,
    }


def convert_nq(ex):
    return {"question": ex["question"], "answers": ex["answer"]}


def preprocess_triviaqa(orig_dir, output_dir, index_dir):
    data, index = {}, {}
    for split in ["train", "dev", "test"]:
        with open(index_dir / ("TQA." + split + ".idx.json"), "r") as fin:
            index[split] = json.load(fin)

    with open(orig_dir / "triviaqa-unfiltered" / "unfiltered-web-train.json") as fin:
        originaltrain = json.load(fin)["Data"]
    with open(orig_dir / "triviaqa-unfiltered" / "unfiltered-web-dev.json") as fin:
        originaldev = json.load(fin)["Data"]

    data["train"] = [convert_triviaqa(originaltrain[k]) for k in index["train"]]
    data["train.64-shot"] = [convert_triviaqa(originaltrain[k]) for k in triviaqa_64shot]
    data["dev"] = [convert_triviaqa(originaltrain[k]) for k in index["dev"]]
    data["test"] = [convert_triviaqa(originaldev[k]) for k in index["test"]]

    for split in data:
        with open(output_dir / (split + ".jsonl"), "w") as fout:
            for ex in data[split]:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")


def preprocess_nq(orig_dir, output_dir, index_dir):
    data, index = {}, {}
    for split in ["train", "dev", "test"]:
        with open(index_dir / ("NQ." + split + ".idx.json"), "r") as fin:
            index[split] = json.load(fin)

    originaltrain, originaldev = [], []
    with open(orig_dir / "NQ-open.dev.jsonl") as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            originaldev.append(example)

    with open(orig_dir / "NQ-open.train.jsonl") as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            originaltrain.append(example)

    data["train"] = [convert_nq(originaltrain[k]) for k in index["train"]]
    data["train.64-shot"] = [convert_nq(originaltrain[k]) for k in nq_64shot]
    data["dev"] = [convert_nq(originaltrain[k]) for k in index["dev"]]
    data["test"] = [convert_nq(originaldev[k]) for k in index["test"]]

    for split in data:
        with open(output_dir / (split + ".jsonl"), "w") as fout:
            for ex in data[split]:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")


def main(args):
    output_dir = Path(args.output_directory)

    index_tar = output_dir / "index.tar"
    index_dir = output_dir / "dataindex"

    original_triviaqa_dir = output_dir / "original_triviaqa"
    triviaqa_dir = output_dir / "triviaqa_data"
    triviaqa_tar = output_dir / "triviaqa_data.tar"
    nq_dir = output_dir / "nq_data"
    original_nq_dir = output_dir / "original_naturalquestions"

    if args.overwrite:
        print("Overwriting NaturalQuestions and TriviaQA")
        download_triviaqa = True
        download_nq = True
    else:
        download_triviaqa = not triviaqa_dir.exists()
        download_nq = not nq_dir.exists()

    if download_triviaqa or download_nq:
        index_url = "https://dl.fbaipublicfiles.com/FiD/data/dataindex.tar.gz"
        maybe_download_file(index_url, index_tar)
        if not os.path.exists(index_dir):
            with tarfile.open(index_tar) as tar:
                tar.extractall(index_dir)

    if download_triviaqa:
        triviaqa_dir.mkdir(parents=True, exist_ok=True)
        original_triviaqa_url = "http://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz"
        maybe_download_file(original_triviaqa_url, triviaqa_tar)
        if not os.path.exists(original_triviaqa_dir):
            with tarfile.open(triviaqa_tar) as tar:
                tar.extractall(original_triviaqa_dir)
        preprocess_triviaqa(original_triviaqa_dir, triviaqa_dir, index_dir)
    else:
        print("TriviaQA data already exists, not overwriting")

    if download_nq:
        nq_dir.mkdir(parents=True, exist_ok=True)
        nq_dev_url = "https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl"
        nq_train_url = "https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.train.jsonl"
        maybe_download_file(nq_dev_url, original_nq_dir / "NQ-open.dev.jsonl")
        maybe_download_file(nq_train_url, original_nq_dir / "NQ-open.train.jsonl")
        preprocess_nq(original_nq_dir, nq_dir, index_dir)
    else:
        print("NaturalQuestions data already exists, not overwriting")

    triviaqa_tar.unlink(missing_ok=True)
    index_tar.unlink(missing_ok=True)
    if original_triviaqa_dir.exists():
        shutil.rmtree(original_triviaqa_dir)
    if original_nq_dir.exists():
        shutil.rmtree(original_nq_dir)
    if index_dir.exists():
        shutil.rmtree(index_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./data/",
        help="Path to the file to which the dataset is written.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite data")
    args = parser.parse_args()
    main(args)