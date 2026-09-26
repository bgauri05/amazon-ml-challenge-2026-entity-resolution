
#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - High precision business entity resolution.

End-to-end pipeline:
  1. Stream-load TSVs.
  2. Unicode/transliteration-aware normalization.
  3. Multi-view blocking (exact name/address, compact forms, rare tokens,
     postal/digit signatures, character q-gram signatures).
  4. Generate positive + hard-negative training pairs from ground truth.
  5. Train LightGBM pair classifier.
  6. Tune a precision-heavy decision policy for macro F0.5.
  7. Predict test matches and write:
       output/matching_results.tsv
       output/candidate_pairs.tsv

IMPORTANT:
- Uses ONLY challenge-provided data. No external lookup.
- The challenge requires every S1 test entity to appear in matching_results.tsv.
- This script does not guarantee any particular leaderboard score. It is
  designed to optimize the stated macro F0.5 objective and must be validated
  on a held-out split before submission.
"""

from __future__ import annotations
import argparse, csv, math, os, re, sqlite3, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio, WRatio, partial_ratio
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

try:
    from unidecode import unidecode
except Exception:
    def unidecode(x):
        return x

SEED = 20260926
ID_COL = "entity_id"
TEXT_COLS = ("business_name", "business_address", "country")
SOURCE_COLS = ["business_name", "business_address", "country"]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

LEGAL = {
    "incorporated", "inc", "corp", "corporation", "co", "company",
    "llc", "ltd", "limited", "llp", "plc", "pvt", "private", "privatelimited",
    "private limited", "pte", "gmbh", "sarl", "sas", "sa", "bv", "ag",
    "l.l.c", "l.l.p"
}

ABBR = {
    # Deliberately generic/open-set; no country-specific postal/state table.
    "street": "st", "st.": "st", "road": "rd", "rd.": "rd",
    "avenue": "ave", "ave.": "ave", "boulevard": "blvd", "blvd.": "blvd",
    "drive": "dr", "dr.": "dr", "lane": "ln", "ln.": "ln",
    "parkway": "pkwy", "highway": "hwy", "suite": "ste", "ste.": "ste",
    "building": "bldg", "floor": "fl", "apartment": "apt",
}

def norm(s: object) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = unidecode(s).lower()
    s = s.replace("&", " and ")
    s = s.replace("@", " at ")
    s = re.sub(r"https?://", " ", s)
    s = re.sub(r"\bwww\.", " ", s)
    s = re.sub(r"\.(com|net|org|in|co|biz|io)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def name_norm(s: object) -> str:
    x = norm(s)
    if not x:
        return ""
    toks = [t for t in x.split() if t not in LEGAL]
    return " ".join(toks)

def compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s)

def addr_norm(s: object) -> str:
    x = norm(s)
    if not x:
        return ""
    toks = [ABBR.get(t, t) for t in x.split()]
    return " ".join(toks)

def tokens(s: str) -> List[str]:
    return s.split() if s else []

def digit_tokens(s: str) -> List[str]:
    return re.findall(r"\d{3,}", s or "")

def grams(s: str, n: int = 3) -> set:
    if not s:
        return set()
    z = f"  {s}  "
    return {z[i:i+n] for i in range(max(0, len(z)-n+1))}

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def containment(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))

def safe_ratio(a: str, b: str) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return ratio(a, b) / 100.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_tsv(path: Path, cols=None) -> pd.DataFrame:
    return pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False,
        usecols=cols, quoting=csv.QUOTE_NONE, on_bad_lines="warn"
    )

def prep(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    out = df.copy()
    out["name_n"] = out["business_name"].map(name_norm)
    out["addr_n"] = out["business_address"].map(addr_norm)
    out["country_n"] = out["country"].map(norm)
    out["name_c"] = out["name_n"].map(compact)
    out["addr_c"] = out["addr_n"].map(compact)
    out["name_t"] = out["name_n"].map(tokens)
    out["addr_t"] = out["addr_n"].map(tokens)
    out["name_d"] = out["name_n"].map(digit_tokens)
    out["addr_d"] = out["addr_n"].map(digit_tokens)
    return out


# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------

def add_postings(index, key, rid):
    if not key:
        return
    index.setdefault(key, []).append(rid)

def build_index(df: pd.DataFrame, max_postings: int = 200):
    """
    Python dictionaries are used only for compact candidate indexes.
    Very common keys are capped to prevent quadratic candidate explosions.
    """
    idx = {
        "name": {}, "name_c": {}, "addr": {}, "addr_c": {},
        "name_tok": {}, "addr_tok": {}, "digit": {}
    }

    # First collect token document frequencies; rare tokens are much safer blocks.
    name_df = Counter()
    addr_df = Counter()
    dig_df = Counter()

    for row in df.itertuples(index=False):
        name_df.update(set(row.name_t))
        addr_df.update(set(row.addr_t))
        dig_df.update(set(row.name_d + row.addr_d))

    for rid, row in enumerate(df.itertuples(index=False)):
        for k, v in (
            ("name", row.name_n), ("name_c", row.name_c),
            ("addr", row.addr_n), ("addr_c", row.addr_c)
        ):
            if v:
                add_postings(idx[k], v, rid)

        # Store only rare/moderate-frequency tokens.
        for t in set(row.name_t):
            if 2 <= len(t) and name_df[t] <= max_postings:
                add_postings(idx["name_tok"], t, rid)
        for t in set(row.addr_t):
            if 2 <= len(t) and addr_df[t] <= max_postings:
                add_postings(idx["addr_tok"], t, rid)
        for d in set(row.name_d + row.addr_d):
            if dig_df[d] <= max_postings:
                add_postings(idx["digit"], d, rid)

    return idx

def candidates_for_row(row, idx, limit=250):
    cand = set()
    exact_keys = [
        ("name", row.name_n), ("name_c", row.name_c),
        ("addr", row.addr_n), ("addr_c", row.addr_c),
    ]
    for typ, key in exact_keys:
        if key:
            cand.update(idx[typ].get(key, []))

    # Rare token blocks. Use the rarest available query tokens first.
    nt = [t for t in set(row.name_t) if len(t) >= 2]
    at = [t for t in set(row.addr_t) if len(t) >= 2]
    for t in sorted(nt, key=lambda x: len(x), reverse=True)[:4]:
        cand.update(idx["name_tok"].get(t, []))
    for t in sorted(at, key=lambda x: len(x), reverse=True)[:4]:
        cand.update(idx["addr_tok"].get(t, []))
    for d in set(row.name_d + row.addr_d):
        cand.update(idx["digit"].get(d, []))

    # Keep deterministic cap. Exact hits are retained first.
    if len(cand) > limit:
        exact = set()
        for typ, key in exact_keys:
            if key:
                exact.update(idx[typ].get(key, []))
        rest = sorted(cand - exact)
        cand = exact | set(rest[:max(0, limit-len(exact))])
    return cand


# ---------------------------------------------------------------------------
# Pair features
# ---------------------------------------------------------------------------

def pair_features(a, b) -> np.ndarray:
    an, bn = a.name_n, b.name_n
    aa, ba = a.addr_n, b.addr_n
    ac, bc = a.addr_c, b.addr_c
    nc, mc = a.name_c, b.name_c
    ant, bnt = set(a.name_t), set(b.name_t)
    aat, bat = set(a.addr_t), set(b.addr_t)
    add, bdd = set(a.addr_d), set(b.addr_d)
    andg, bndg = grams(an), grams(bn)
    aadg, badg = grams(aa), grams(ba)

    ntok_inter = len(ant & bnt)
    atok_inter = len(aat & bat)
    dtok_inter = len(add & bdd)

    f = [
        float(a.country_n == b.country_n and a.country_n != ""),
        float(an == bn and an != ""),
        float(nc == mc and nc != ""),
        float(aa == ba and aa != ""),
        float(ac == bc and ac != ""),
        safe_ratio(an, bn),
        token_sort_ratio(an, bn) / 100.0,
        token_set_ratio(an, bn) / 100.0,
        WRatio(an, bn) / 100.0 if an and bn else 0.0,
        partial_ratio(an, bn) / 100.0 if an and bn else 0.0,
        jaccard(andg, bndg),
        containment(andg, bndg),
        safe_ratio(aa, ba),
        token_sort_ratio(aa, ba) / 100.0,
        token_set_ratio(aa, ba) / 100.0,
        WRatio(aa, ba) / 100.0 if aa and ba else 0.0,
        partial_ratio(aa, ba) / 100.0 if aa and ba else 0.0,
        jaccard(aadg, badg),
        containment(aadg, badg),
        jaccard(ant, bnt),
        containment(ant, bnt),
        jaccard(aat, bat),
        containment(aat, bat),
        float(ntok_inter),
        float(atok_inter),
        float(dtok_inter),
        float(len(an)), float(len(bn)), float(abs(len(an)-len(bn))),
        float(len(aa)), float(len(ba)), float(abs(len(aa)-len(ba))),
    ]
    return np.asarray(f, dtype=np.float32)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def read_gt(path: Path) -> Dict[str, set]:
    gt = {}
    for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).itertuples(index=False):
        ids = set(x for x in str(ch.matched_entity_ids).split(",") if x)
        gt[ch.source1_entity_id] = ids
    return gt


# ---------------------------------------------------------------------------
# Candidate generation + pair dataset
# ---------------------------------------------------------------------------

def make_pair_dataset(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: Dict[str, set],
    s1_indices: Sequence[int],
    neg_per_pos: int = 5,
    candidate_limit: int = 250,
):
    """
    Train on true pairs + hard negatives generated by the exact same blocking
    mechanism used at inference. This makes validation much more realistic
    than random negatives.
    """
    idx2 = build_index(s2)
    idx3 = build_index(s3)

    rows = []
    labels = []

    for ii in s1_indices:
        a = s1.iloc[ii]
        true = gt.get(a.entity_id, set())

        c2 = candidates_for_row(a, idx2, candidate_limit)
        c3 = candidates_for_row(a, idx3, candidate_limit)

        # positives
        pos = []
        for j in c2:
            if s2.iloc[j].entity_id in true:
                pos.append((s2.iloc[j], 1))
        for j in c3:
            if s3.iloc[j].entity_id in true:
                pos.append((s3.iloc[j], 1))

        for b, y in pos:
            rows.append(pair_features(a, b)); labels.append(y)

        # hard negatives: candidate records not in truth.
        neg = []
        for j in c2:
            if s2.iloc[j].entity_id not in true:
                neg.append(s2.iloc[j])
        for j in c3:
            if s3.iloc[j].entity_id not in true:
                neg.append(s3.iloc[j])

        # Prefer the most lexically similar negatives.
        if len(neg) > neg_per_pos * max(1, len(pos)):
            neg.sort(
                key=lambda b: (
                    0.55 * safe_ratio(a.name_n, b.name_n) +
                    0.45 * safe_ratio(a.addr_n, b.addr_n)
                ),
                reverse=True,
            )
            neg = neg[:neg_per_pos * max(1, len(pos))]

        for b in neg:
            rows.append(pair_features(a, b)); labels.append(0)

    if not rows:
        raise RuntimeError("No training pairs were generated. Check paths and ground truth.")
    return np.vstack(rows), np.asarray(labels, dtype=np.int8), idx2, idx3


# ---------------------------------------------------------------------------
# Metric + decision policy
# ---------------------------------------------------------------------------

def f05_one(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    if not pred:
        return 0.0
    if not true:
        return 0.0
    tp = len(pred & true)
    p = tp / len(pred)
    r = tp / len(true)
    if p == 0 or r == 0:
        return 0.0
    return 1.25 * p * r / (0.25 * p + r)

def macro_f05(all_pred: Dict[str, set], gt: Dict[str, set], ids: Sequence[str]) -> float:
    return float(np.mean([f05_one(all_pred.get(i, set()), gt.get(i, set())) for i in ids]))

def calibrate_threshold(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: Dict[str, set],
    model,
    val_indices: Sequence[int],
    idx2, idx3,
    candidate_limit=250,
):
    """
    Tune for macro F0.5. We deliberately use high thresholds and a margin
    rule because false merges are disproportionately expensive.
    """
    cache = []
    ids = []
    for ii in val_indices:
        a = s1.iloc[ii]
        ids.append(a.entity_id)
        true = gt.get(a.entity_id, set())
        cands = [(s2.iloc[j], f"S2:{j}") for j in candidates_for_row(a, idx2, candidate_limit)]
        cands += [(s3.iloc[j], f"S3:{j}") for j in candidates_for_row(a, idx3, candidate_limit)]
        X = np.vstack([pair_features(a, b) for b, _ in cands]) if cands else np.empty((0,32),np.float32)
        probs = model.predict_proba(X)[:,1] if len(X) else np.empty(0)
        cache.append((a, cands, probs))

    best = (-1.0, None)
    # Candidate thresholds concentrate around the high precision regime.
    thresholds = np.unique(np.r_[
        np.linspace(0.50, 0.99, 50),
        np.linspace(0.990, 0.9999, 30)
    ])
    margins = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
    for th in thresholds:
        for margin in margins:
            pred = {}
            for a, cands, probs in cache:
                chosen = []
                if len(probs):
                    order = np.argsort(-probs)
                    for k in order:
                        p = float(probs[k])
                        if p < th:
                            break
                        if chosen and p < float(probs[chosen[0]]) - margin:
                            break
                        chosen.append(k)
                pred[a.entity_id] = {
                    cands[k][0].entity_id for k in chosen
                }
            score = macro_f05(pred, gt, ids)
            if score > best[0]:
                best = (score, (float(th), float(margin)))
    print(f"[calibration] best held-out macro F0.5={best[0]:.6f}, threshold={best[1][0]:.5f}, margin={best[1][1]:.3f}")
    return best[1]


# ---------------------------------------------------------------------------
# Full inference
# ---------------------------------------------------------------------------

def predict_all(
    s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
    model, idx2, idx3, threshold, margin, candidate_limit=250
):
    match_rows = []
    cand_rows = []

    for n, a in enumerate(s1.itertuples(index=False), 1):
        c2 = candidates_for_row(a, idx2, candidate_limit)
        c3 = candidates_for_row(a, idx3, candidate_limit)
        cands = [(s2.iloc[j], j, "S2") for j in c2]
        cands += [(s3.iloc[j], j, "S3") for j in c3]

        if cands:
            X = np.vstack([pair_features(a, b) for b, _, _ in cands])
            probs = model.predict_proba(X)[:,1]
            order = np.argsort(-probs)
        else:
            probs = np.empty(0)
            order = []

        chosen = []
        if len(probs):
            top = float(probs[order[0]])
            for k in order:
                p = float(probs[k])
                if p < threshold:
                    break
                if chosen and p < top - margin:
                    break
                chosen.append(k)

        match_ids = []
        for k in chosen:
            b, _, src = cands[k]
            match_ids.append(b.entity_id)

        candidate_ids = [b.entity_id for b, _, _ in cands]

        match_rows.append((a.entity_id, ",".join(dict.fromkeys(match_ids))))
        cand_rows.append((a.entity_id, ",".join(dict.fromkeys(candidate_ids))))

        if n % 10000 == 0:
            print(f"[inference] {n:,}/{len(s1):,}")

    return match_rows, cand_rows


def write_outputs(outdir: Path, match_rows, cand_rows):
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir/"matching_results.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        w.writerows(match_rows)

    with open(outdir/"candidate_pairs.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "candidate_entity_ids"])
        w.writerows(cand_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root containing train/ and test/")
    ap.add_argument("--out", default="output")
    ap.add_argument("--candidate-limit", type=int, default=250)
    ap.add_argument("--neg-per-pos", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--num-leaves", type=int, default=96)
    ap.add_argument("--n-estimators", type=int, default=1800)
    args = ap.parse_args()

    root = Path(args.data)
    train = root/"train"
    test = root/"test"

    print("[1/7] Loading training data...")
    tr1 = prep(read_tsv(train/"train_source1.tsv"))
    tr2 = prep(read_tsv(train/"train_source2.tsv"))
    tr3 = prep(read_tsv(train/"train_source3.tsv"))
    gt = read_gt(train/"train_ground_truth.tsv")

    rng = np.random.default_rng(SEED)
    all_idx = np.arange(len(tr1))
    rng.shuffle(all_idx)
    cut = int(len(all_idx) * (1 - args.val_frac))
    train_idx, val_idx = all_idx[:cut], all_idx[cut:]

    print(f"train S1={len(train_idx):,}, validation S1={len(val_idx):,}")

    print("[2/7] Building hard-negative training pairs...")
    Xtr, ytr, idx2, idx3 = make_pair_dataset(
        tr1, tr2, tr3, gt, train_idx,
        neg_per_pos=args.neg_per_pos,
        candidate_limit=args.candidate_limit
    )
    print("pair matrix:", Xtr.shape, "positive rate:", ytr.mean())

    print("[3/7] Training LightGBM...")
    # Class weighting is intentionally moderate; the decision threshold is
    # separately tuned for the precision-heavy metric.
    pos = max(1, int(ytr.sum()))
    neg = max(1, int(len(ytr)-pos))
    clf = LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=0.035,
        num_leaves=args.num_leaves,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_alpha=0.25,
        reg_lambda=1.5,
        random_state=SEED,
        n_jobs=max(1, os.cpu_count() - 1),
        verbosity=-1,
    )
    # No random holdout inside pair training: S1-level validation is handled
    # explicitly below, which matches the competition's macro entity metric.
    clf.fit(Xtr, ytr, sample_weight=np.where(ytr==1, 1.0, 0.75))

    print("[4/7] Calibrating macro-F0.5 decision policy...")
    threshold, margin = calibrate_threshold(
        tr1, tr2, tr3, gt, clf, val_idx, idx2, idx3,
        candidate_limit=args.candidate_limit
    )

    print("[5/7] Loading test...")
    te1 = prep(read_tsv(test/"test_source1.tsv"))
    te2 = prep(read_tsv(test/"test_source2.tsv"))
    te3 = prep(read_tsv(test/"test_source3.tsv"))

    print("[6/7] Building test candidate indexes...")
    te_idx2 = build_index(te2)
    te_idx3 = build_index(te3)

    print("[7/7] Generating final outputs...")
    match_rows, cand_rows = predict_all(
        te1, te2, te3, clf, te_idx2, te_idx3,
        threshold, margin, candidate_limit=args.candidate_limit
    )
    write_outputs(Path(args.out), match_rows, cand_rows)

    print("\nDONE")
    print("threshold:", threshold)
    print("margin:", margin)
    print("matching:", Path(args.out)/"matching_results.tsv")
    print("candidates:", Path(args.out)/"candidate_pairs.tsv")


if __name__ == "__main__":
    main()
