"""
f_beta_scorer.py

Computes the competition metric locally: macro-averaged F_0.5 over Source 1 entities.

F_0.5 = (1.25 * P * R) / (0.25 * P + R)

Per-entity rule:
- If ground truth is empty (singleton) AND prediction is empty -> F0.5 = 1.0
- If ground truth is empty AND prediction is non-empty         -> F0.5 = 0.0
- If ground truth is non-empty AND prediction is empty         -> F0.5 = 0.0 (recall = 0)
- Otherwise compute precision/recall normally over set overlap.

Usage:
    python -m src.evaluation.f_beta_scorer \
        --predictions output/val_predictions.tsv \
        --ground-truth data/processed/val_ground_truth.tsv
"""

import argparse
import csv
from pathlib import Path


def _read_id_list_tsv(path):
    """
    Reads a two-column TSV: source1_entity_id \t comma_separated_ids (may be empty).
    Returns dict: {source1_entity_id: set(ids)}
    """
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            if len(row) == 1:
                s1_id, id_list_str = row[0], ""
            else:
                s1_id, id_list_str = row[0], row[1]
            ids = set(x.strip() for x in id_list_str.split(",") if x.strip())
            result[s1_id] = ids
    return result


def per_entity_f_beta(pred_ids: set, true_ids: set, beta: float = 0.5) -> float:
    """Compute F_beta for a single Source 1 entity."""
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids and pred_ids:
        return 0.0
    if true_ids and not pred_ids:
        return 0.0

    tp = len(pred_ids & true_ids)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)

    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    f_beta = (1 + beta_sq) * precision * recall / denom
    return f_beta


def score(predictions_path: str, ground_truth_path: str, beta: float = 0.5, verbose: bool = True):
    preds = _read_id_list_tsv(predictions_path)
    truth = _read_id_list_tsv(ground_truth_path)

    missing = set(truth.keys()) - set(preds.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} Source 1 entities from ground truth are missing in predictions. "
            f"Example missing ids: {list(missing)[:5]}"
        )

    scores = []
    per_entity = {}
    for s1_id, true_ids in truth.items():
        pred_ids = preds.get(s1_id, set())
        f = per_entity_f_beta(pred_ids, true_ids, beta=beta)
        scores.append(f)
        per_entity[s1_id] = f

    macro_f = sum(scores) / len(scores) if scores else 0.0

    if verbose:
        n_singletons = sum(1 for t in truth.values() if not t)
        n_multi = len(truth) - n_singletons
        print(f"Entities scored: {len(truth)} (singletons: {n_singletons}, with matches: {n_multi})")
        print(f"Macro F_{beta}: {macro_f:.4f}")

    return macro_f, per_entity


def main():
    parser = argparse.ArgumentParser(description="Score predictions against ground truth using macro F_0.5.")
    parser.add_argument("--predictions", required=True, help="Path to predictions TSV (source1_entity_id, matched_entity_ids)")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth TSV (source1_entity_id, matched_entity_ids)")
    parser.add_argument("--beta", type=float, default=0.5)
    args = parser.parse_args()

    score(args.predictions, args.ground_truth, beta=args.beta, verbose=True)


if __name__ == "__main__":
    main()
