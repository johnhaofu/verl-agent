"""Generate dummy parquet files for Spider training.

verl-agent's standard `prepare.py` downloads geometry3k from HuggingFace
just to copy its row schema (data_source / prompt / ability / extra_info).
The actual env data lives in the live envs created by env_manager. This
self-contained script bypasses HF entirely (DSW often can't reach HF) and
emits the same shape directly.

Usage:
    python -m examples.data_preprocess.prepare_spider_dummy \
        --train_data_size 8 --val_data_size 128
"""
import argparse
import os
import pandas as pd


def make_dummy(n: int, split: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": split, "index": i},
        }
        for i in range(n)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="text", choices=["text"])
    parser.add_argument("--local_dir", default=os.path.expanduser("~/data/verl-agent/"))
    parser.add_argument("--train_data_size", type=int, default=8)
    parser.add_argument("--val_data_size", type=int, default=128)
    args = parser.parse_args()

    out_dir = os.path.join(args.local_dir, args.mode)
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.parquet")
    test_path = os.path.join(out_dir, "test.parquet")

    make_dummy(args.train_data_size, "train").to_parquet(train_path)
    make_dummy(args.val_data_size, "test").to_parquet(test_path)

    print(f"wrote {train_path} (n={args.train_data_size})")
    print(f"wrote {test_path} (n={args.val_data_size})")


if __name__ == "__main__":
    main()
