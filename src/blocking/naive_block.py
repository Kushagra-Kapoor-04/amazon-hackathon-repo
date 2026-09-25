"""
naive_block.py

Baseline blocking strategy: generate candidates using normalized-name token
overlap, optionally filtered by country. This is intentionally simple —
it exists to get a working end-to-end pipeline and first submission fast.
Replace / augment with tfidf_lsh.py and phonetic_block.py for better recall.
"""

import csv
from collections import defaultdict

from src.preprocessing.normalize import normalize_name, normalize_country


def load_records(path):
    """Load a source TSV into a list of dicts."""
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records.append(row)
    return records


def build_token_index(records):
    """
    Build an inverted index: token -> list of entity_ids.
    Tokens come from normalized business_name.
    """
    index = defaultdict(set)
    norm_cache = {}
    for r in records:
        norm_name = normalize_name(r.get("business_name", ""))
        norm_cache[r["entity_id"]] = {
            "norm_name": norm_name,
            "country": normalize_country(r.get("country", "")),
        }
        for token in norm_name.split():
            if len(token) >= 3:  # skip very short/common tokens
                index[token].add(r["entity_id"])
    return index, norm_cache


def generate_candidates(source1_records, source2_records, source3_records,
                         top_k=30, use_country_filter=True,
                         unknown_country_fallback=True):
    """
    For each Source 1 record, find candidate Source 2 / Source 3 records
    that share at least one normalized-name token (and optionally country).

    Returns: dict {source1_entity_id: [candidate_entity_id, ...]}
    """
    idx2, cache2 = build_token_index(source2_records)
    idx3, cache3 = build_token_index(source3_records)

    known_countries = set()
    for r in source1_records + source2_records + source3_records:
        c = normalize_country(r.get("country", ""))
        if c:
            known_countries.add(c)

    results = {}
    for r in source1_records:
        s1_id = r["entity_id"]
        norm_name = normalize_name(r.get("business_name", ""))
        s1_country = normalize_country(r.get("country", ""))
        tokens = [t for t in norm_name.split() if len(t) >= 3]

        candidate_scores = defaultdict(int)

        for token in tokens:
            for cand_id in idx2.get(token, ()):  # Source 2 hits
                candidate_scores[cand_id] += 1
            for cand_id in idx3.get(token, ()):  # Source 3 hits
                candidate_scores[cand_id] += 1

        # Country filter: only applied when country is known for both sides,
        # unless the record's country is unseen (e.g. France at test time),
        # in which case we fall back to no country filter for that entity.
        country_is_novel = s1_country not in known_countries if unknown_country_fallback else False

        if use_country_filter and s1_country and not country_is_novel:
            filtered = {}
            for cand_id, score in candidate_scores.items():
                cand_country = cache2.get(cand_id, cache3.get(cand_id, {})).get("country", "")
                if cand_country == s1_country or not cand_country:
                    filtered[cand_id] = score
            candidate_scores = filtered

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [cand_id for cand_id, _ in ranked[:top_k]]
        results[s1_id] = top_candidates

    return results


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    prefix = args.split

    s1 = load_records(paths[f"{prefix}_source1"])
    s2 = load_records(paths[f"{prefix}_source2"])
    s3 = load_records(paths[f"{prefix}_source3"])

    block_cfg = cfg["blocking"]
    candidates = generate_candidates(
        s1, s2, s3,
        top_k=block_cfg["top_k_per_entity"],
        use_country_filter=block_cfg["use_country_filter"],
        unknown_country_fallback=block_cfg["unknown_country_fallback"],
    )

    out_path = paths["candidate_pairs_out"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id, cand_ids in candidates.items():
            writer.writerow([s1_id, ",".join(cand_ids)])

    print(f"Wrote candidates for {len(candidates)} Source 1 entities to {out_path}")
