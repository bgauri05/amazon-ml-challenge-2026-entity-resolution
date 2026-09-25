#!/usr/bin/env python3

"""
AMAZON ML CHALLENGE 2026
GPU-Accelerated Business Entity Resolution  (FIXED)

Changes vs. the original version:
    1. TargetStore now also builds a numeric-token inverted index
       (street numbers / PIN codes / any digit run in the address)
       per (source, country). At retrieval time, candidates found
       via this index are UNIONED with the FAISS ANN candidates.
       This rescues true matches that the char-ngram embedding
       cannot find on its own -- most importantly cross-script /
       transliterated business names, where character overlap is
       near zero but digit strings in the address are identical
       regardless of script.
    2. Default --dim, --top-k and --ef-search raised, since the
       observed candidate recall (~44%) was the bottleneck on the
       final F0.5 score, not the classifier.
    3. XGBoost is trained AND scored on device="cpu" unconditionally.
       Training on cuda while scoring cpu-resident numpy arrays
       (as in the original) triggers XGBoost's slow "mismatched
       devices" fallback on every batch. For ~10^5 rows x 22
       features, GPU transfer overhead exceeds any compute benefit,
       so cpu-cpu is both correct and faster in practice.

Architecture:
    1. Character-level TF-IDF
    2. Truncated SVD
    3. CUDA-accelerated embedding projection
    4. FAISS HNSW approximate nearest-neighbor retrieval
    5. Numeric-token inverted-index blocking (NEW - recall rescue)
    6. Similarity feature engineering
    7. XGBoost classification (CPU)
    8. Macro F0.5 threshold optimization
    9. Full test inference
    10. Official submission validation

Install:
    python -m pip install numpy pandas scipy scikit-learn
    python -m pip install faiss-cpu rapidfuzz joblib xgboost torch

Quick experiment:
    python model.py --sample 10000 --top-k 40 --skip-test

Full execution:
    python model.py --sample 80000 --top-k 40
"""

import argparse
import csv
import gc
import logging
import re
import sqlite3
import subprocess
import sys
import unicodedata

from collections import defaultdict
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from rapidfuzz import fuzz

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


# ============================================================
# CONFIGURATION
# ============================================================

LOG = logging.getLogger("entity_resolution")

SEED = 42

COLS = ["entity_id", "business_name", "business_address", "country"]

# Minimum length of a digit run to count as a blocking key.
# 3+ digits filters out noise like single house-number digits
# reused across many unrelated addresses (low precision, high
# collision rate) while keeping street numbers / PIN codes / zip
# codes, which are usually >=3 digits and fairly discriminative.
MIN_NUMERIC_TOKEN_LEN = 3

# Cap on how many numeric-blocked candidates we add per query,
# per (source, country) group. This is a blocking stage, not the
# final classifier -- keep it generous enough to catch true
# matches, but bounded so candidate_pairs.tsv doesn't explode and
# precision at inference isn't diluted with junk.
MAX_NUMERIC_CANDIDATES = 15

FEATURE_NAMES = [
    "embedding",
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_weighted",
    "addr_ratio",
    "addr_token_sort",
    "addr_token_set",
    "name_jaccard",
    "addr_jaccard",
    "exact_name",
    "exact_addr",
    "number_overlap",
    "both_numbers",
    "number_disagree",
    "name_length_gap",
    "addr_length_gap",
    "missing_name_left",
    "missing_name_right",
    "missing_addr_left",
    "missing_addr_right",
    "country_equal",
]


# ============================================================
# 1. TEXT NORMALIZATION
# ============================================================

def norm(s):
    """Normalize business names, addresses and countries. Unicode-aware,
    country-independent, no hardcoded country/state lookups."""

    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def record(row):
    """Convert a source row into a normalized tuple."""
    return (str(row[0]), norm(row[1]), norm(row[2]), norm(row[3]))


def text_of(r):
    """Generate retrieval text. Business names are repeated to increase
    their influence during vectorization relative to the address."""
    return f"{r[1]} {r[1]} {r[2]}"


def numeric_tokens(addr):
    """Extract digit runs (len >= MIN_NUMERIC_TOKEN_LEN) from a
    normalized address string. These are used as a script-independent
    blocking key: even when a business name is transliterated into a
    completely different script (near-zero char n-gram overlap), the
    digits in a street number or PIN code are usually rendered in
    Latin numerals and remain identical across sources."""
    return {t for t in re.findall(r"\d+", addr) if len(t) >= MIN_NUMERIC_TOKEN_LEN}


# ============================================================
# 2. DATA LOADING
# ============================================================

def source_path(data, split, source):
    return data / split / f"{split}_source{source}.tsv"


def iter_records(path, chunksize=50000):
    """Stream large TSV files in chunks; avoid loading whole files."""
    reader = pd.read_csv(
        path, sep="\t", usecols=COLS, dtype=str,
        keep_default_na=False, chunksize=chunksize,
    )
    for chunk in reader:
        for row in chunk[COLS].itertuples(index=False, name=None):
            yield record(row)


def read_truth(path):
    """Load training ground truth: S1 id -> set of matching S2/S3 ids."""
    truth = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            source1_id = row["source1_entity_id"]
            matched = row["matched_entity_ids"]
            truth[source1_id] = set(filter(None, matched.split(",")))
    return truth


def sample_records(path, n, seed=SEED):
    """Reservoir sampling without loading the whole file into memory."""
    rng = np.random.default_rng(seed)
    out = []
    for i, r in enumerate(iter_records(path)):
        if i < n:
            out.append(r)
        else:
            j = int(rng.integers(i + 1))
            if j < n:
                out[j] = r
    return out


# ============================================================
# 3. GPU-ACCELERATED SEMANTIC ENCODER
# ============================================================

class Encoder:

    def __init__(self, dim, device):
        self.dim = dim
        self.device = device
        self.vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), min_df=2,
            max_features=150000, sublinear_tf=True, dtype=np.float32,
        )
        self.svd = TruncatedSVD(n_components=dim, n_iter=4, random_state=SEED)
        self.projection = None

    def fit(self, paths, per_source, seed=SEED):
        """Fit TF-IDF + SVD on sampled training records (CPU), then
        move the fitted projection matrix onto the target device."""
        samples = []
        for i, path in enumerate(paths):
            sampled = sample_records(path, per_source, seed + i)
            samples.extend(text_of(r) for r in sampled)

        LOG.info("Fitting TF-IDF/SVD on %s sampled records", len(samples))

        matrix = self.vectorizer.fit_transform(samples)
        self.svd.fit(matrix)

        del matrix, samples
        gc.collect()

        self.projection = torch.from_numpy(
            self.svd.components_.T.copy()
        ).to(self.device)

    @torch.no_grad()
    def transform(self, records, batch_size=2000):
        """Generate normalized dense embeddings."""
        output = []

        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            texts = [text_of(r) for r in batch]
            sparse = self.vectorizer.transform(texts).tocoo()

            if self.device.type == "cuda":
                ij = np.vstack((sparse.row, sparse.col)).astype(np.int64)
                indices = torch.from_numpy(ij).to(self.device)
                values = torch.from_numpy(sparse.data).to(self.device)

                tensor = torch.sparse_coo_tensor(
                    indices, values, size=sparse.shape, device=self.device
                ).coalesce()

                dense = torch.sparse.mm(tensor, self.projection).float()
                dense = torch.nn.functional.normalize(dense, dim=1)

                arr = dense.cpu().numpy()
                del tensor, dense, indices, values
            else:
                arr = normalize(
                    sparse.tocsr() @ self.svd.components_.T
                ).astype(np.float32)

            output.append(np.ascontiguousarray(arr, dtype=np.float32))

        if not output:
            return np.empty((0, self.dim), dtype=np.float32)

        return np.concatenate(output)


# ============================================================
# 4. FAISS TARGET INDEX + NUMERIC BLOCKING INDEX
# ============================================================

class TargetStore:
    """Disk-backed target metadata, FAISS HNSW indexes, and a numeric-
    token inverted index, all keyed by (source, country). Country
    labels are discovered dynamically -- no hardcoded country/state
    tables, per the open-set requirement."""

    def __init__(self, db_path, dim, m=16, ef_construction=100, ef_search=128):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS targets (
                id TEXT PRIMARY KEY,
                name TEXT,
                addr TEXT,
                country TEXT
            )
            """
        )

        self.indexes = {}
        self.ids = {}
        # Numeric inverted index: (source, country, number_token) -> [ids]
        self.numeric_index = defaultdict(list)

        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def build(self, data, split, encoder, chunk_size=20000):
        """Build FAISS indexes + numeric inverted index from S2/S3."""

        counts = defaultdict(int)

        for source in (2, 3):
            path = source_path(data, split, source)
            reader = pd.read_csv(
                path, sep="\t", usecols=["country"], dtype=str,
                keep_default_na=False, chunksize=100000,
            )
            for chunk in reader:
                values = chunk["country"].map(norm).value_counts()
                for country, count in values.items():
                    counts[(source, country)] += int(count)

        for key, count in counts.items():
            index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            self.indexes[key] = index
            self.ids[key] = []
            LOG.info("Index %s: %s records", key, count)

        for source in (2, 3):
            path = source_path(data, split, source)
            buf = []
            processed = 0

            for r in iter_records(path):
                buf.append(r)
                if len(buf) >= chunk_size:
                    self._add(buf, source, encoder)
                    processed += len(buf)
                    if processed % 200000 < chunk_size:
                        LOG.info("Source %s indexed %s", source, processed)
                    buf.clear()

            if buf:
                self._add(buf, source, encoder)

            self.conn.commit()

    def _add(self, records, source, encoder):
        """Generate embeddings, add to FAISS, and populate the numeric
        inverted index."""

        vectors = encoder.transform(records)

        groups = defaultdict(list)
        for i, r in enumerate(records):
            groups[r[3]].append(i)

        self.conn.executemany(
            "INSERT OR REPLACE INTO targets VALUES (?,?,?,?)", records
        )

        for country, positions in groups.items():
            key = (source, country)
            v = np.ascontiguousarray(vectors[positions], dtype=np.float32)
            self.indexes[key].add(v)
            self.ids[key].extend(records[i][0] for i in positions)

        for i, r in enumerate(records):
            rid, _name, addr, country = r
            for token in numeric_tokens(addr):
                self.numeric_index[(source, country, token)].append(rid)

        self.conn.commit()

    def get_many(self, ids):
        """Retrieve target metadata in SQLite batches."""
        out = {}
        ids = list(dict.fromkeys(ids))

        for start in range(0, len(ids), 400):
            part = ids[start:start + 400]
            if not part:
                continue
            placeholders = ",".join("?" * len(part))
            query = (
                "SELECT id,name,addr,country FROM targets WHERE id IN ("
                + placeholders + ")"
            )
            for r in self.conn.execute(query, part):
                out[r[0]] = r

        return out

    def _numeric_candidates(self, r, source):
        """Look up candidates sharing a numeric address token with
        record r, for a given target source. Capped and deduplicated."""
        _rid, _name, addr, country = r
        found = []
        for token in numeric_tokens(addr):
            found.extend(self.numeric_index.get((source, country, token), ()))
            if len(found) >= MAX_NUMERIC_CANDIDATES:
                break
        # De-dup while preserving order, then cap.
        return list(dict.fromkeys(found))[:MAX_NUMERIC_CANDIDATES]

    def retrieve(self, records, encoder, top_k):
        """Retrieve nearest neighbors for each Source 1 record from both
        target sources, via FAISS ANN, UNIONED with numeric-token
        blocking candidates. The numeric-token union is what rescues
        true matches the embedding alone cannot find (e.g. transliterated
        names with near-zero character overlap but identical address
        digit strings)."""

        vectors = encoder.transform(records)
        output = [[] for _ in records]

        grouped = defaultdict(list)
        for i, r in enumerate(records):
            grouped[r[3]].append(i)

        for country, positions in grouped.items():
            q = np.ascontiguousarray(vectors[positions], dtype=np.float32)

            for source in (2, 3):
                key = (source, country)

                if key in self.indexes and self.indexes[key].ntotal > 0:
                    index = self.indexes[key]
                    count = min(top_k, index.ntotal)
                    scores, labels = index.search(q, count)
                    ids = self.ids[key]

                    for local_i, original_i in enumerate(positions):
                        for label, score in zip(labels[local_i], scores[local_i]):
                            if label < 0:
                                continue
                            output[original_i].append((ids[int(label)], float(score)))

                # Numeric-token blocking union. No retrieval score is
                # available for these, so they're tagged with score 0.0
                # (the classifier's "missing-retrieval" convention,
                # consistent with training_arrays below) -- the fuzzy
                # / jaccard / exact-match features still let the
                # classifier confirm or reject them.
                for original_i in positions:
                    r = records[original_i]
                    existing = {cid for cid, _ in output[original_i]}
                    for cid in self._numeric_candidates(r, source):
                        if cid not in existing:
                            output[original_i].append((cid, 0.0))
                            existing.add(cid)

        return output

    def close(self):
        self.conn.close()


# ============================================================
# 5. SIMILARITY FEATURE ENGINEERING
# ============================================================

def jaccard(a, b):
    a, b = set(a.split()), set(b.split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def features(a, b, similarity):
    """Extract numerical similarity features: semantic similarity, fuzzy
    name/address similarity, token overlap, address number agreement,
    missing-value indicators, country agreement."""

    an, aa, ac = a[1:]
    bn, ba, bc = b[1:]

    na = set(re.findall(r"\d+", aa))
    nb = set(re.findall(r"\d+", ba))

    return [
        similarity,
        fuzz.ratio(an, bn) / 100,
        fuzz.token_sort_ratio(an, bn) / 100,
        fuzz.token_set_ratio(an, bn) / 100,
        fuzz.WRatio(an, bn) / 100,
        fuzz.ratio(aa, ba) / 100,
        fuzz.token_sort_ratio(aa, ba) / 100,
        fuzz.token_set_ratio(aa, ba) / 100,
        jaccard(an, bn),
        jaccard(aa, ba),
        float(bool(an) and an == bn),
        float(bool(aa) and aa == ba),
        float(bool(na & nb)),
        float(bool(na) and bool(nb)),
        float(bool(na) and bool(nb) and na != nb),
        abs(len(an) - len(bn)),
        abs(len(aa) - len(ba)),
        float(not an),
        float(not bn),
        float(not aa),
        float(not ba),
        float(ac == bc),
    ]


# ============================================================
# 6. CANDIDATE RECALL
# ============================================================

def candidate_recall(records, retrieved, truth):
    """Recall of the blocking / candidate-generation stage. This is the
    ceiling on the final F0.5 -- anything missed here can never be
    predicted correctly downstream."""
    total = found = 0
    for r, candidates in zip(records, retrieved):
        gt = truth.get(r[0], set())
        predicted = {cid for cid, _ in candidates}
        total += len(gt)
        found += len(gt & predicted)

    score = found / total if total else 0.0
    LOG.info("Candidate recall: %.5f (%s/%s)", score, found, total)
    return score


# ============================================================
# 7. TRAINING PAIR GENERATION
# ============================================================

def training_arrays(records, candidates, truth, store, max_neg=8):
    """Build supervised training pairs: positives from ground truth,
    hard negatives from top retrieved-but-unmatched candidates."""

    X, y = [], []

    for r, retrieved in zip(records, candidates):
        actual = truth.get(r[0], set())
        scores = dict(retrieved)
        sorted_candidates = sorted(retrieved, key=lambda p: -p[1])

        negative = [cid for cid, _ in sorted_candidates if cid not in actual][:max_neg]
        selected = list(actual) + negative
        targets = store.get_many(selected)

        for cid in selected:
            if cid not in targets:
                continue
            # A positive missed by ANN/numeric blocking has no
            # retrieval score; set to zero as a missing-retrieval flag.
            similarity = scores.get(cid, 0.0)
            X.append(features(r, targets[cid], similarity))
            y.append(int(cid in actual))

    if not X or len(set(y)) != 2:
        raise RuntimeError("Training requires both positive and negative examples.")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8)


# ============================================================
# 8. XGBOOST CLASSIFIER (CPU)
# ============================================================

def fit_model(X, y):
    """Train XGBoost on CPU.

    FIX: the original trained with device="cuda" but scored with plain
    numpy arrays on CPU, which forces XGBoost's slow "mismatched
    devices" DMatrix fallback on every prediction call (visible in the
    run log). For a dataset this size (~10^5 rows x 22 features) the
    GPU offers no meaningful speedup anyway -- the bottleneck is
    Python-level feature engineering, not tree building. Training and
    scoring on CPU consistently removes the fallback and is faster
    end-to-end.
    """

    model = xgb.XGBClassifier(
        n_estimators=450,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.9,
        tree_method="hist",
        device="cpu",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=8,
    )

    LOG.info("Training XGBoost on cpu")
    model.fit(X, y, verbose=False)
    return model


# ============================================================
# 9. BATCHED INFERENCE
# ============================================================

def score_batch(records, candidates, store, model):
    """Generate features and predict match probabilities for one batch."""

    X, meta = [], []
    all_ids = [cid for row in candidates for cid, _ in row]
    targets = store.get_many(all_ids)
    out = {r[0]: [] for r in records}

    for r, row in zip(records, candidates):
        for cid, sim in row:
            if cid not in targets:
                continue
            X.append(features(r, targets[cid], sim))
            meta.append((r[0], cid))

    if X:
        matrix = np.asarray(X, dtype=np.float32)
        probs = model.predict_proba(matrix)[:, 1]
        for (rid, cid), p in zip(meta, probs):
            out[rid].append((cid, float(p)))

    return out


# ============================================================
# 10. OFFICIAL MACRO F0.5 EVALUATION
# ============================================================

def entity_f05(actual, predicted):
    """F0.5 for one Source 1 entity. Correctly predicted singletons = 1.0."""
    if not actual and not predicted:
        return 1.0
    if not actual or not predicted:
        return 0.0

    tp = len(actual & predicted)
    if not tp:
        return 0.0

    return 1.25 * tp / (0.25 * len(actual) + len(predicted))


# ============================================================
# 11. THRESHOLD OPTIMIZATION
# ============================================================

def choose_threshold(scored, truth):
    """Optimize classification threshold using validation macro F0.5."""

    best_score = float("-inf")
    best_threshold = 0.75
    thresholds = np.arange(0.30, 0.991, 0.025)

    for threshold in thresholds:
        scores = []
        for rid, pairs in scored.items():
            predicted = {cid for cid, p in pairs if p >= threshold}
            actual = truth.get(rid, set())
            scores.append(entity_f05(actual, predicted))

        score = float(np.mean(scores))
        LOG.info("Threshold %.3f | Macro F0.5 %.5f", threshold, score)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    LOG.info("Best validation F0.5: %.5f", best_score)
    LOG.info("Selected threshold: %.3f", best_threshold)
    return best_threshold


# ============================================================
# 12. TEST INFERENCE AND SUBMISSION GENERATION
# ============================================================

def write_test(data, encoder, model, threshold, args):
    """Run inference over the complete test dataset. Writes
    matching_results.tsv and candidate_pairs.tsv."""

    LOG.info("Building TEST target indexes.")

    db = args.work / "test_targets.sqlite"
    if db.exists():
        db.unlink()

    store = TargetStore(db, args.dim, ef_search=args.ef_search)
    store.build(data, "test", encoder, chunk_size=args.index_batch)

    match_path = args.output / "matching_results.tsv"
    cand_path = args.output / "candidate_pairs.tsv"

    with (
        match_path.open("w", encoding="utf-8", newline="") as mf,
        cand_path.open("w", encoding="utf-8", newline="") as cf,
    ):
        mw = csv.writer(mf, delimiter="\t", lineterminator="\n")
        cw = csv.writer(cf, delimiter="\t", lineterminator="\n")

        mw.writerow(["source1_entity_id", "matched_entity_ids"])
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])

        batch = []
        count = 0

        def process(records):
            retrieved = store.retrieve(records, encoder, args.top_k)
            scored = score_batch(records, retrieved, store, model)

            for r, pairs in zip(records, retrieved):
                rid = r[0]
                cids = list(dict.fromkeys(cid for cid, _ in pairs))
                matched = list(dict.fromkeys(
                    cid for cid, p in scored[rid] if p >= threshold
                ))

                cw.writerow([rid, ",".join(cids)])
                mw.writerow([rid, ",".join(matched)])

        test_path = source_path(data, "test", 1)

        for r in iter_records(test_path):
            batch.append(r)
            if len(batch) >= args.query_batch:
                process(batch)
                count += len(batch)
                LOG.info("Test S1 processed: %s", count)
                batch.clear()

        if batch:
            process(batch)
            count += len(batch)

        LOG.info("Wrote %s test S1 rows", count)

    store.close()

    # --------------------------------------------------------
    # OFFICIAL VALIDATION
    # --------------------------------------------------------

    validator = data.parent / "utils" / "validate_submission.py"
    if not validator.exists():
        validator = data.parent / "validate_submission.py"

    if validator.exists():
        LOG.info("Running official validator.")
        subprocess.run(
            [
                sys.executable, str(validator),
                "--matching", str(match_path),
                "--candidate", str(cand_path),
                "--test-dir", str(data / "test"),
            ],
            check=True,
        )
    else:
        LOG.warning("Validator not found. Run the official validator manually.")


# ============================================================
# 13. MAIN PIPELINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="GPU-accelerated Amazon ML Business Entity Resolution"
    )

    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=30000)
    # top-k raised: with numeric blocking now unioned in, a slightly
    # larger ANN top-k plus the numeric candidates gives a materially
    # higher recall ceiling without exploding candidate_pairs.tsv size.
    parser.add_argument("--top-k", type=int, default=40)
    # dim raised: 64 was likely too small to separate near-duplicate
    # noisy names once averaged with address text; 96 gives more
    # headroom at a modest cost in FAISS memory/build time.
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--svd-samples", type=int, default=60000)
    parser.add_argument("--query-batch", type=int, default=1000)
    parser.add_argument("--index-batch", type=int, default=10000)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument(
        "--skip-test", action="store_true",
        help="Train and validate only. Skip full test inference.",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Disable GPU acceleration for the encoder."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # --------------------------------------------------------
    # DIRECTORY CONFIGURATION
    # --------------------------------------------------------

    base = Path(__file__).resolve().parent

    if args.data is None:
        if (base / "dataset").exists():
            args.data = base / "dataset"
        else:
            args.data = base / "student_resource" / "dataset"

    args.output = args.data.parent / "output"
    args.work = args.data.parent / "artifacts_gpu"

    args.output.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # GPU DETECTION (encoder only -- XGBoost always runs on CPU now)
    # --------------------------------------------------------

    cuda_available = torch.cuda.is_available()
    device = torch.device("cpu") if (args.cpu or not cuda_available) else torch.device("cuda")

    LOG.info("PyTorch device (encoder): %s", device)
    LOG.info("CUDA available: %s", cuda_available)
    LOG.info("XGBoost device: cpu (fixed -- avoids device-mismatch fallback)")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 2**30
        LOG.info("GPU: %s", gpu_name)
        LOG.info("VRAM: %.2f GiB", gpu_memory)
    else:
        LOG.warning("CUDA acceleration disabled for the encoder. Running on CPU.")

    # --------------------------------------------------------
    # TRAINING PATHS
    # --------------------------------------------------------

    paths = [source_path(args.data, "train", s) for s in (1, 2, 3)]

    # --------------------------------------------------------
    # SEMANTIC ENCODER
    # --------------------------------------------------------

    LOG.info("Training semantic encoder.")
    encoder = Encoder(args.dim, device)
    encoder.fit(paths, args.svd_samples)

    # Temporarily remove the CUDA tensor before serializing.
    projection = encoder.projection
    encoder.projection = None
    joblib.dump(encoder, args.work / "encoder.joblib")
    encoder.projection = projection

    # --------------------------------------------------------
    # LOAD GROUND TRUTH
    # --------------------------------------------------------

    truth = read_truth(args.data / "train" / "train_ground_truth.tsv")

    # --------------------------------------------------------
    # SAMPLE TRAINING REFERENCES
    # --------------------------------------------------------

    LOG.info("Sampling %s training references.", args.sample)
    sampled = sample_records(paths[0], args.sample)
    train, valid = train_test_split(sampled, test_size=0.2, random_state=SEED)

    # --------------------------------------------------------
    # BUILD TRAINING INDEXES (FAISS + numeric)
    # --------------------------------------------------------

    db = args.work / "train_targets.sqlite"
    if db.exists():
        db.unlink()

    store = TargetStore(db, args.dim, ef_search=args.ef_search)

    LOG.info("Building training FAISS + numeric-token indexes.")
    store.build(args.data, "train", encoder, chunk_size=args.index_batch)

    # --------------------------------------------------------
    # TRAINING CANDIDATES
    # --------------------------------------------------------

    LOG.info("Retrieving training candidates.")
    train_candidates = store.retrieve(train, encoder, args.top_k)
    candidate_recall(train, train_candidates, truth)

    # --------------------------------------------------------
    # TRAINING FEATURES
    # --------------------------------------------------------

    LOG.info("Generating training features.")
    X, y = training_arrays(train, train_candidates, truth, store)

    LOG.info("Training pairs: %s", len(y))
    LOG.info("Positive pairs: %s", int(y.sum()))

    # --------------------------------------------------------
    # CLASSIFIER TRAINING (CPU)
    # --------------------------------------------------------

    model = fit_model(X, y)
    model.save_model(str(args.work / "xgboost.json"))

    del X, y, train_candidates
    gc.collect()

    # --------------------------------------------------------
    # VALIDATION RETRIEVAL
    # --------------------------------------------------------

    LOG.info("Retrieving validation candidates.")
    valid_candidates = store.retrieve(valid, encoder, args.top_k)
    candidate_recall(valid, valid_candidates, truth)

    # --------------------------------------------------------
    # VALIDATION INFERENCE
    # --------------------------------------------------------

    scored = {}
    for start in range(0, len(valid), args.query_batch):
        batch_records = valid[start:start + args.query_batch]
        batch_candidates = valid_candidates[start:start + args.query_batch]
        scored.update(score_batch(batch_records, batch_candidates, store, model))

    # --------------------------------------------------------
    # THRESHOLD OPTIMIZATION
    # --------------------------------------------------------

    threshold = choose_threshold(scored, truth)
    joblib.dump({"threshold": threshold}, args.work / "threshold.joblib")

    # --------------------------------------------------------
    # MEMORY CLEANUP
    # --------------------------------------------------------

    del scored, valid_candidates, train, valid, truth, sampled
    store.close()
    del store
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # TEST INFERENCE
    # --------------------------------------------------------

    if not args.skip_test:
        write_test(args.data, encoder, model, threshold, args)

    LOG.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()