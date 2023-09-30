import os

import wget

BASE_URL = "https://dl.fbaipublicfiles.com/atlas"


def maybe_download_file(source, target):
    if not os.path.exists(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        print(f"Downloading {source} to {target}")
        wget.download(source, out=str(target))
        print()


def get_s3_path(path):
    return f"{BASE_URL}/{path}"


def get_download_path(output_dir, path):
    return os.path.join(output_dir, path)