"""
infer.py

Day-1 baseline: a rule-based scorer over pairwise features (no trained model yet).
Swap `rule_based_score` for a real trained model's `.predict_proba()` once
train.py produces one — the pipeline.py call site stays the same.

Scoring is intentionally precision-biased: "when unsure, do not merge."
"""

from src.features.pairwise_features import compute_pair_features


def rule_based_score(record1: dict, record2: dict) -> float:
    """
    Cheap weighted-average scorer used as a placeholder for the trained model.
    Returns a score in [0, 1]; higher = more likely a true match.
    """
    f = compute_pair_features(record1, record2)

    if f["name1_empty"] or f["name2_empty"]:
        return 0.0

    name_score = (
        0.35 * f["name_jaccard_token"] +
        0.25 * f["name_levenshtein_ratio"] +
        0.20 * f["name_token_sort_ratio"] +
        0.20 * f["name_core_exact_match"]
    )
    address_score = (
        0.5 * f["address_jaccard_token"] +
        0.5 * f["address_levenshtein_ratio"]
    )

    score = 0.65 * name_score + 0.35 * address_score

    # Small bonus for exact name match or matching country; small penalty if
    # country is known on both sides and disagrees (strong negative signal).
    if f["name_exact_match"]:
        score = min(1.0, score + 0.1)
    if f["country_known_flag"] and not f["country_match_flag"]:
        score *= 0.5

    return max(0.0, min(1.0, score))


def load_trained_model(model_path: str):
    """
    Placeholder loader for the real model (train.py). Return None until a
    trained model artifact exists; pipeline.py falls back to rule_based_score.
    """
    import os
    if not os.path.exists(model_path):
        return None
    import joblib
    return joblib.load(model_path)


def model_score(model, record1: dict, record2: dict) -> float:
    """Score a pair using a trained model (expects a scikit-learn-like API)."""
    feats = compute_pair_features(record1, record2)
    feature_order = sorted(feats.keys())
    X = [[feats[k] for k in feature_order]]
    proba = model.predict_proba(X)[0][1]
    return float(proba)
