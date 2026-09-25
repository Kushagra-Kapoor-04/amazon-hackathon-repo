"""
pipeline.py

End-to-end orchestrator: raw source TSVs -> candidate_pairs.tsv -> matching_results.tsv

Day-1 baseline: naive blocking + rule-based scoring.
As train.py produces a real trained model, this will pick it up automatically
(see load_trained_model in modeling/infer.py) — no other changes needed.

Usage:
    python -m src.pipeline --config configs/config.yaml --split test
"""

import argparse
import csv
from pathlib import Path

import yaml

from src.blocking.naive_block import load_records, generate_candidates
from src.modeling.infer import rule_based_score, load_trained_model, model_score


def write_id_list_tsv(path, header, id_map):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for s1_id, ids in id_map.items():
            writer.writerow([s1_id, ",".join(ids)])


def run_pipeline(cfg: dict, split: str = "test"):
    paths = cfg["paths"]
    block_cfg = cfg["blocking"]
    decision_cfg = cfg["decision"]

    s1 = load_records(paths[f"{split}_source1"])
    s2 = load_records(paths[f"{split}_source2"])
    s3 = load_records(paths[f"{split}_source3"])

    s2_by_id = {r["entity_id"]: r for r in s2}
    s3_by_id = {r["entity_id"]: r for r in s3}
    s1_by_id = {r["entity_id"]: r for r in s1}

    print(f"[{split}] Loaded {len(s1)} Source1, {len(s2)} Source2, {len(s3)} Source3 records.")

    # --- Stage 1: Blocking ---
    print("Running blocking...")
    candidates = generate_candidates(
        s1, s2, s3,
        top_k=block_cfg["top_k_per_entity"],
        use_country_filter=block_cfg["use_country_filter"],
        unknown_country_fallback=block_cfg["unknown_country_fallback"],
    )

    candidate_pairs_path = paths["candidate_pairs_out"]
    write_id_list_tsv(candidate_pairs_path, ["source1_entity_id", "candidate_entity_ids"], candidates)
    total_candidates = sum(len(v) for v in candidates.values())
    print(f"Wrote {total_candidates} candidate pairs across {len(candidates)} entities -> {candidate_pairs_path}")

    # --- Stage 2: Scoring ---
    print("Scoring candidates...")
    model_path = cfg.get("model", {}).get("artifact_path", "data/processed/model.joblib")
    model = load_trained_model(model_path)
    if model is not None:
        print(f"Loaded trained model from {model_path}")
    else:
        print("No trained model found — using rule-based placeholder scorer.")

    threshold = decision_cfg["score_threshold"]
    abstain_below = decision_cfg["abstain_if_top_score_below"]

    matches = {}
    for s1_id, cand_ids in candidates.items():
        record1 = s1_by_id[s1_id]
        scored = []
        for cand_id in cand_ids:
            record2 = s2_by_id.get(cand_id) or s3_by_id.get(cand_id)
            if record2 is None:
                continue
            score = model_score(model, record1, record2) if model is not None else rule_based_score(record1, record2)
            scored.append((cand_id, score))

        if not scored:
            matches[s1_id] = []
            continue

        top_score = max(s for _, s in scored)
        if top_score < abstain_below:
            matches[s1_id] = []
        else:
            matches[s1_id] = [cid for cid, s in scored if s >= threshold]

    matching_results_path = paths["matching_results_out"]
    write_id_list_tsv(matching_results_path, ["source1_entity_id", "matched_entity_ids"], matches)
    total_matches = sum(len(v) for v in matches.values())
    n_singletons_predicted = sum(1 for v in matches.values() if not v)
    print(f"Wrote {total_matches} matches ({n_singletons_predicted} predicted singletons) -> {matching_results_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", choices=["train", "test"], default="test",
                         help="Which source files to run inference over.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_pipeline(cfg, split=args.split)


if __name__ == "__main__":
    main()
