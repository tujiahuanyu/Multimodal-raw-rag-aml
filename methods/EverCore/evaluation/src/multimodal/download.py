"""Download the public Mem-Gallery data with resumable bounded retries."""

import argparse
import os
from pathlib import Path
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args(argv)
    if arguments.attempts < 1 or arguments.workers < 1:
        parser.error("--attempts and --workers must be positive")
    os.environ["HF_ENDPOINT"] = arguments.endpoint
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    from huggingface_hub import snapshot_download

    for attempt in range(arguments.attempts):
        try:
            destination = snapshot_download(
                repo_id="Ethan-Bei/Mem-Gallery",
                repo_type="dataset",
                revision=arguments.revision,
                local_dir=str(arguments.data_root),
                endpoint=arguments.endpoint,
                allow_patterns=[
                    "data/dialog/*.json",
                    "data/image/**/*.png",
                    "data/image/**/*.jpg",
                    "data/image/**/*.jpeg",
                    "data/image/**/*.webp",
                ],
                ignore_patterns=["**/.DS_Store"],
                max_workers=arguments.workers,
            )
            print(f"Dataset ready: {destination}", flush=True)
            return
        except Exception as error:
            print(
                f"Download attempt {attempt + 1}/{arguments.attempts} failed: {type(error).__name__}",
                flush=True,
            )
            if attempt + 1 == arguments.attempts:
                raise
            time.sleep(5)


if __name__ == "__main__":
    main()
