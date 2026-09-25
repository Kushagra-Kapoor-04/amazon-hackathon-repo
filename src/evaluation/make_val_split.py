"""
make_val_split.py

Splits train_source1 entities (and their ground truth) into a train-subset and
a held-out validation subset. Source 2 / Source 3 records are NOT split — they
stay whole, since a Source 1 entity's true matches could be anywhere in them.

Produces:
    data/processed/val_source1.tsv
    data/processed/val_ground_truth.tsv
    data/processed/train_subset_source1.tsv
    data/processed/train_subset_ground_truth.tsv

Usage:
    python -m src.evaluation.make_val_split --config configs/config.yaml
"""

import argparse
import csv
import random
from pathlib import Path

import yaml


def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = [row for row in reader if row]
    return header, rows


def write_tsv(path, header, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    val_cfg = cfg["validation"]
    random.seed(val_cfg["random_seed"])

    s1_header, s1_rows = read_tsv(paths["train_source1"])
    gt_header, gt_rows = read_tsv(paths["train_ground_truth"])

    gt_by_id = {row[0]: (row[1] if len(row) > 1 else "") for row in gt_rows}

    s1_ids = [row[0] for row in s1_rows]

    if val_cfg.get("stratify_on_singleton", True):
        singleton_ids = [sid for sid in s1_ids if not gt_by_id.get(sid, "").strip()]
        multi_ids = [sid for sid in s1_ids if gt_by_id.get(sid, "").strip()]
        random.shuffle(singleton_ids)
        random.shuffle(multi_ids)

        n_val_singleton = int(len(singleton_ids) * val_cfg["val_fraction"])
        n_val_multi = int(len(multi_ids) * val_cfg["val_fraction"])

        val_ids = set(singleton_ids[:n_val_singleton] + multi_ids[:n_val_multi])
    else:
        shuffled = s1_ids[:]
        random.shuffle(shuffled)
        n_val = int(len(shuffled) * val_cfg["val_fraction"])
        val_ids = set(shuffled[:n_val])

    val_s1_rows = [row for row in s1_rows if row[0] in val_ids]
    train_subset_s1_rows = [row for row in s1_rows if row[0] not in val_ids]

    val_gt_rows = [row for row in gt_rows if row[0] in val_ids]
    train_subset_gt_rows = [row for row in gt_rows if row[0] not in val_ids]

    out_dir = paths["val_split_dir"]
    write_tsv(f"{out_dir}/val_source1.tsv", s1_header, val_s1_rows)
    write_tsv(f"{out_dir}/val_ground_truth.tsv", gt_header, val_gt_rows)
    write_tsv(f"{out_dir}/train_subset_source1.tsv", s1_header, train_subset_s1_rows)
    write_tsv(f"{out_dir}/train_subset_ground_truth.tsv", gt_header, train_subset_gt_rows)

    print(f"Total Source 1 train entities: {len(s1_rows)}")
    print(f"Validation entities: {len(val_s1_rows)}")
    print(f"Train-subset entities: {len(train_subset_s1_rows)}")
    print(f"Written to: {out_dir}/")


if __name__ == "__main__":
    main()
