"""
data_inspection.py

Stage 1 - Data Audit.
Loads all challenge TSV files (train + test), reports:
  - Row counts
  - Column names
  - Missing-value rates
  - Duplicate entity IDs
  - Country distributions
  - Representative samples
  - Ground-truth statistics (zero / single / multiple matches per Source-1 entity)
  - Match-count distribution

Output is written to experiments/data_inspection.txt and also printed to stdout.

Usage (from repo root):
    python -m src.data_inspection --config configs/config.yaml
"""

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

class Report:
    """Accumulates report lines and writes them to a file and stdout."""

    def __init__(self, out_path, also_print=True):
        self.lines = []
        self.out_path = out_path
        self.also_print = also_print

    def __call__(self, *args, **kwargs):
        sep = kwargs.get("sep", " ")
        line = sep.join(str(a) for a in args)
        self.lines.append(line)
        if self.also_print:
            print(line)

    def save(self):
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
        if self.also_print:
            print(f"\n[data_inspection] Report saved to: {self.out_path}")


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------

def load_tsv(path):
    """Load a TSV into (header, rows). Returns (None, None) if file missing."""
    if not os.path.isfile(path):
        return None, None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        header = list(reader.fieldnames or [])
    return header, rows


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------

def inspect_source_file(label, path, rpt):
    """Inspect a source TSV and append stats to rpt."""
    rpt("")
    rpt("=" * 70)
    rpt("  FILE: " + label)
    rpt("  PATH: " + path)
    rpt("=" * 70)

    if not os.path.isfile(path):
        rpt("  *** FILE NOT FOUND ***")
        return None

    header, rows = load_tsv(path)
    rpt("  Row count     : {:,}".format(len(rows)))
    rpt("  Columns       : " + str(header))

    if not rows:
        rpt("  (file is empty)")
        return None

    MISSING = {"", "nan", "none", "null", "na", "n/a"}

    rpt("")
    rpt("  Missing-value counts:")
    for col in header:
        n_missing = sum(
            1 for r in rows if str(r.get(col, "") or "").strip().lower() in MISSING
        )
        pct = 100 * n_missing / len(rows)
        rpt("    {:<30} {:>7,}  ({:.1f}%)".format(col, n_missing, pct))

    if "entity_id" in header:
        ids = [r["entity_id"] for r in rows]
        id_counts = Counter(ids)
        dups = {eid: cnt for eid, cnt in id_counts.items() if cnt > 1}
        rpt("")
        rpt("  Duplicate entity_ids : {:,}".format(len(dups)))
        if dups:
            rpt("  Sample duplicates    : " + str(list(dups.items())[:5]))
    else:
        rpt("")
        rpt("  [WARNING] No 'entity_id' column found!")

    if "country" in header:
        countries = Counter(str(r.get("country", "") or "").strip() for r in rows)
        rpt("")
        rpt("  Country distribution:")
        for country, cnt in countries.most_common():
            pct = 100 * cnt / len(rows)
            cname = country if country else "(empty)"
            rpt("    {:<25} {:>7,}  ({:.1f}%)".format(cname, cnt, pct))
    else:
        rpt("")
        rpt("  [WARNING] No 'country' column found!")

    if "entity_id" in header:
        prefixes = Counter(
            r["entity_id"].split("-")[0] if "-" in r["entity_id"] else "??"
            for r in rows
        )
        rpt("")
        rpt("  entity_id prefix distribution: " + str(dict(prefixes)))

    rpt("")
    rpt("  Sample rows (first 3):")
    for i, row in enumerate(rows[:3]):
        rpt("    [{}] {}".format(i, dict(row)))

    return rows


def inspect_ground_truth(label, path, rpt):
    """Inspect ground-truth TSV and report match-count distributions."""
    rpt("")
    rpt("=" * 70)
    rpt("  FILE: " + label)
    rpt("  PATH: " + path)
    rpt("=" * 70)

    if not os.path.isfile(path):
        rpt("  *** FILE NOT FOUND ***")
        return None

    header, rows = load_tsv(path)
    rpt("  Row count : {:,}".format(len(rows)))
    rpt("  Columns   : " + str(header))

    if not rows:
        rpt("  (file is empty)")
        return None

    MISSING = {"", "nan", "none", "null", "na", "n/a"}
    s1_col = "source1_entity_id"
    m_col = "matched_entity_ids"

    if s1_col not in header or m_col not in header:
        rpt("  [WARNING] Expected columns '{}' and '{}' not found.".format(s1_col, m_col))
        rpt("           Actual columns: " + str(header))
        return rows

    match_counts = []
    source2_hits = 0
    source3_hits = 0
    all_matched_ids = []

    for row in rows:
        raw = str(row.get(m_col, "") or "").strip()
        if raw.lower() in MISSING:
            ids = []
        else:
            ids = [x.strip() for x in raw.split(",") if x.strip()]
        match_counts.append(len(ids))
        for mid in ids:
            if mid.startswith("S2-"):
                source2_hits += 1
            elif mid.startswith("S3-"):
                source3_hits += 1
        all_matched_ids.extend(ids)

    total = len(match_counts)
    n_zero = sum(1 for c in match_counts if c == 0)
    n_one = sum(1 for c in match_counts if c == 1)
    n_multi = sum(1 for c in match_counts if c > 1)

    rpt("")
    rpt("  Match statistics over {:,} Source-1 entities:".format(total))
    rpt("    Singletons (0 matches)     : {:>7,}  ({:.1f}%)".format(n_zero, 100 * n_zero / total))
    rpt("    Exactly 1 match            : {:>7,}  ({:.1f}%)".format(n_one, 100 * n_one / total))
    rpt("    2+ matches                 : {:>7,}  ({:.1f}%)".format(n_multi, 100 * n_multi / total))
    rpt("")
    rpt("    Total matched IDs          : {:>7,}".format(len(all_matched_ids)))
    rpt("    Of which S2- IDs           : {:>7,}".format(source2_hits))
    rpt("    Of which S3- IDs           : {:>7,}".format(source3_hits))

    count_dist = Counter(min(c, 10) for c in match_counts)
    rpt("")
    rpt("  Distribution of match count (cap 10+):")
    for k in range(11):
        n = count_dist.get(k, 0)
        bar = "#" * min(n * 40 // max(total, 1) + (1 if n > 0 else 0), 40)
        label_k = "{}+".format(k) if k == 10 else str(k)
        rpt("    {:>4} matches : {:>7,}  {}".format(label_k, n, bar))

    non_zero = [c for c in match_counts if c > 0]
    if non_zero:
        rpt("")
        rpt("  Among entities with >=1 match:")
        rpt("    Min  matches : {}".format(min(non_zero)))
        rpt("    Max  matches : {}".format(max(non_zero)))
        rpt("    Mean matches : {:.2f}".format(sum(non_zero) / len(non_zero)))
        rpt("    Median       : {}".format(sorted(non_zero)[len(non_zero) // 2]))

    s1_ids = [row[s1_col] for row in rows]
    s1_id_counts = Counter(s1_ids)
    dups = {eid: cnt for eid, cnt in s1_id_counts.items() if cnt > 1}
    rpt("")
    rpt("  Duplicate source1_entity_id rows : {:,}".format(len(dups)))
    if dups:
        rpt("  Sample: " + str(list(dups.items())[:5]))

    rpt("")
    rpt("  Sample rows (first 5):")
    for i, row in enumerate(rows[:5]):
        rpt("    [{}] {}".format(i, dict(row)))

    return rows


# ---------------------------------------------------------------------------
# Cross-file consistency checks
# ---------------------------------------------------------------------------

def cross_checks(train_s1, train_s2, train_s3,
                 test_s1, test_s2, test_s3,
                 gt_rows, rpt):
    rpt("")
    rpt("=" * 70)
    rpt("  CROSS-FILE CONSISTENCY CHECKS")
    rpt("=" * 70)

    def id_set(rows, col="entity_id"):
        if rows is None:
            return set()
        return {r.get(col, "") for r in rows}

    train_s1_ids = id_set(train_s1)
    train_s2_ids = id_set(train_s2)
    train_s3_ids = id_set(train_s3)
    test_s1_ids = id_set(test_s1)
    test_s2_ids = id_set(test_s2)
    test_s3_ids = id_set(test_s3)

    if gt_rows is not None and train_s1 is not None:
        gt_s1_ids = {r.get("source1_entity_id", "") for r in gt_rows}
        only_in_gt = gt_s1_ids - train_s1_ids
        only_in_s1 = train_s1_ids - gt_s1_ids
        rpt("")
        rpt("  Ground-truth vs train_source1:")
        rpt("    S1 IDs in GT but not in train_source1 : {:,}".format(len(only_in_gt)))
        rpt("    S1 IDs in train_source1 but not in GT : {:,}".format(len(only_in_s1)))
        if only_in_gt:
            rpt("    Sample (GT-only)  : " + str(list(only_in_gt)[:5]))
        if only_in_s1:
            rpt("    Sample (S1-only)  : " + str(list(only_in_s1)[:5]))

    pairs = [
        ("train_s1", train_s1_ids, "train_s2", train_s2_ids),
        ("train_s1", train_s1_ids, "train_s3", train_s3_ids),
        ("train_s2", train_s2_ids, "train_s3", train_s3_ids),
        ("test_s1", test_s1_ids, "test_s2", test_s2_ids),
        ("test_s1", test_s1_ids, "test_s3", test_s3_ids),
        ("test_s2", test_s2_ids, "test_s3", test_s3_ids),
    ]
    for name_a, ids_a, name_b, ids_b in pairs:
        if ids_a and ids_b:
            overlap = ids_a & ids_b
            rpt("")
            rpt("  entity_id overlap {} intersect {} : {:,}".format(name_a, name_b, len(overlap)))
            if overlap:
                rpt("    Sample: " + str(list(overlap)[:5]))

    for name_a, ids_a, name_b, ids_b in [
        ("train_s1", train_s1_ids, "test_s1", test_s1_ids),
        ("train_s2", train_s2_ids, "test_s2", test_s2_ids),
        ("train_s3", train_s3_ids, "test_s3", test_s3_ids),
    ]:
        if ids_a and ids_b:
            overlap = ids_a & ids_b
            rpt("")
            rpt("  Train/test ID overlap {} intersect {} : {:,}".format(name_a, name_b, len(overlap)))
            if overlap:
                rpt("    [WARNING] Sample: " + str(list(overlap)[:5]))

    if gt_rows is not None and (train_s2 is not None or train_s3 is not None):
        MISSING = {"", "nan", "none", "null", "na", "n/a"}
        all_matched = set()
        for row in gt_rows:
            raw = str(row.get("matched_entity_ids", "") or "").strip()
            if raw.lower() not in MISSING:
                for mid in raw.split(","):
                    mid = mid.strip()
                    if mid:
                        all_matched.add(mid)
        known_s23 = train_s2_ids | train_s3_ids
        orphan_matches = all_matched - known_s23
        rpt("")
        rpt("  Matched IDs in GT absent from S2/S3 files : {:,}".format(len(orphan_matches)))
        if orphan_matches:
            rpt("    Sample orphans: " + str(list(orphan_matches)[:10]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 data audit - Business Entity Resolution Challenge"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="experiments/data_inspection.txt")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    rpt = Report(args.output, also_print=True)

    rpt("=" * 70)
    rpt("  ML CHALLENGE 2026 -- DATA INSPECTION REPORT")
    rpt("  Config : " + args.config)
    rpt("=" * 70)

    train_s1 = inspect_source_file("TRAIN SOURCE 1", paths["train_source1"], rpt)
    train_s2 = inspect_source_file("TRAIN SOURCE 2", paths["train_source2"], rpt)
    train_s3 = inspect_source_file("TRAIN SOURCE 3", paths["train_source3"], rpt)
    gt = inspect_ground_truth("TRAIN GROUND TRUTH", paths["train_ground_truth"], rpt)
    test_s1 = inspect_source_file("TEST SOURCE 1", paths["test_source1"], rpt)
    test_s2 = inspect_source_file("TEST SOURCE 2", paths["test_source2"], rpt)
    test_s3 = inspect_source_file("TEST SOURCE 3", paths["test_source3"], rpt)

    cross_checks(train_s1, train_s2, train_s3,
                 test_s1, test_s2, test_s3, gt, rpt)

    rpt("")
    rpt("=" * 70)
    rpt("  SUMMARY")
    rpt("=" * 70)

    def safe_len(rows):
        return "{:,}".format(len(rows)) if rows is not None else "FILE NOT FOUND"

    rpt("  train_source1.tsv  : " + safe_len(train_s1) + " rows")
    rpt("  train_source2.tsv  : " + safe_len(train_s2) + " rows")
    rpt("  train_source3.tsv  : " + safe_len(train_s3) + " rows")
    rpt("  train_ground_truth : " + safe_len(gt) + " rows")
    rpt("  test_source1.tsv   : " + safe_len(test_s1) + " rows")
    rpt("  test_source2.tsv   : " + safe_len(test_s2) + " rows")
    rpt("  test_source3.tsv   : " + safe_len(test_s3) + " rows")

    rpt("")
    rpt("  All checks complete.")

    rpt.save()


if __name__ == "__main__":
    main()
