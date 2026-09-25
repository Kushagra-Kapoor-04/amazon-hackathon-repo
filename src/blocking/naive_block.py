"""
naive_block.py

Baseline blocking strategy: generate candidates using normalized-name token
overlap, optionally filtered by country.

IMPORTANT: legal-suffix / boilerplate tokens (limited, private, ltd, llc,
and their transliterations e.g. लिमिटेड, प्राइवेट, or French sarl/sas) are
extremely common and useless for blocking — indexing them naively causes an
O(n^2) blowup where a single token maps to hundreds of thousands of records.
This version strips legal suffixes before tokenizing for the index, AND caps
any token whose document frequency exceeds `max_doc_frequency` as a safety
net against any other unexpectedly common word (including non-English ones
this stopword list doesn't anticipate).
"""

import csv
from collections import defaultdict

from src.preprocessing.normalize import normalize_name, normalize_country, strip_legal_suffix

# Extra boilerplate tokens to exclude from blocking keys, beyond the legal
# suffixes already stripped by strip_legal_suffix(). Includes transliterations
# and common multi-country legal/corporate boilerplate seen in this dataset.
STOPWORDS = {
    "limited", "private", "ltd", "llc", "inc", "pvt", "corp", "corporation",
    "company", "co", "llp", "group", "holdings", "enterprises", "services",
    "partners", "center", "and", "sarl", "sas", "sa",
    "लिमिटेड", "प्राइवेट",  # Hindi transliterations of Limited / Private
    "(india)",
}


def load_records(path):
    """Load a source TSV into a list of dicts."""
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records.append(row)
    return records


def _blocking_tokens(business_name: str):
    """
    Tokens used as blocking keys: normalized, legal-suffix stripped,
    stopwords removed, length >= 3.
    """
    core_name = strip_legal_suffix(business_name)  # drops trailing legal suffix
    tokens = [t for t in core_name.split() if len(t) >= 3 and t not in STOPWORDS]
    return tokens


def build_token_index(records, max_doc_frequency: int = 500):
    """
    Build an inverted index: token -> set of entity_ids, skipping stopwords
    and any token whose document frequency exceeds max_doc_frequency (a
    safety net against unexpectedly common tokens blowing up candidate sets).
    """
    raw_index = defaultdict(set)
    norm_cache = {}

    for r in records:
        norm_name = normalize_name(r.get("business_name", ""))
        norm_cache[r["entity_id"]] = {
            "norm_name": norm_name,
            "country": normalize_country(r.get("country", "")),
        }
        for token in _blocking_tokens(r.get("business_name", "")):
            raw_index[token].add(r["entity_id"])

    # Drop tokens that are still too common after stopword removal.
    dropped = 0
    index = {}
    for token, ids in raw_index.items():
        if len(ids) > max_doc_frequency:
            dropped += 1
            continue
        index[token] = ids

    if dropped:
        print(f"[blocking] Dropped {dropped} high-frequency tokens "
              f"(doc frequency > {max_doc_frequency}) to prevent candidate blowup.")

    return index, norm_cache


def generate_candidates(source1_records, source2_records, source3_records,
                         top_k=30, use_country_filter=True,
                         unknown_country_fallback=True,
                         max_doc_frequency=500):
    """
    For each Source 1 record, find candidate Source 2 / Source 3 records
    that share at least one blocking token (and optionally country).

    Returns: dict {source1_entity_id: [candidate_entity_id, ...]}
    """
    idx2, cache2 = build_token_index(source2_records, max_doc_frequency=max_doc_frequency)
    idx3, cache3 = build_token_index(source3_records, max_doc_frequency=max_doc_frequency)

    known_countries = set()
    for r in source1_records + source2_records + source3_records:
        c = normalize_country(r.get("country", ""))
        if c:
            known_countries.add(c)

    results = {}
    total = len(source1_records)
    for i, r in enumerate(source1_records):
        if i > 0 and i % 5000 == 0:
            print(f"[blocking] Processed {i}/{total} Source 1 entities...")

        s1_id = r["entity_id"]
        s1_country = normalize_country(r.get("country", ""))
        tokens = _blocking_tokens(r.get("business_name", ""))

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
        max_doc_frequency=block_cfg.get("max_doc_frequency", 500),
    )

    out_path = paths["candidate_pairs_out"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id, cand_ids in candidates.items():
            writer.writerow([s1_id, ",".join(cand_ids)])

    print(f"Wrote candidates for {len(candidates)} Source 1 entities to {out_path}")
