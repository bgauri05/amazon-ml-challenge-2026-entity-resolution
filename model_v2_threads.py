
#!/usr/bin/env python3
"""
AMAZON ML CHALLENGE 2026
GPU-Accelerated Business Entity Resolution
Architecture:
    1. Character-level TF-IDF
    2. Truncated SVD
    3. CUDA-accelerated embedding projection
    4. FAISS HNSW approximate nearest-neighbor retrieval
    5. Similarity feature engineering
    6. GPU XGBoost classification
    7. Macro F0.5 threshold optimization
    8. Full test inference
    9. Official submission validation
Hardware:
    NVIDIA RTX 4060
    8 GB VRAM
    32 GB System RAM
The pipeline uses:
    GPU: Embedding projection and XGBoost
    CPU: TF-IDF, SVD fitting, FAISS HNSW, fuzzy matching
    Disk: SQLite target metadata
No external business data or entity-resolution APIs.
Install:
    python -m pip install numpy pandas scipy scikit-learn
    python -m pip install faiss-cpu rapidfuzz joblib xgboost torch
Quick experiment:
    python model.py --sample 10000 --top-k 20 --skip-test
Full execution:
    python model.py --sample 80000 --top-k 40
"""
import argparse
import csv
import gc
import logging
import os
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
COLS = [
    "entity_id",
    "business_name",
    "business_address",
    "country"
]
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
    "canonical_name_equal",
    "canonical_address_equal",
    "name_containment",
    "address_containment"
]
# ============================================================
# 1. TEXT NORMALIZATION
# ============================================================
def norm(s):
    """
    Normalize business names, addresses and countries.
    Unicode-aware and country-independent.
    """
    s = unicodedata.normalize(
        "NFKD",
        str(s or "").lower()
    )
    s = "".join(
        c for c in s
        if not unicodedata.combining(c)
    )
    s = s.replace("&", " and ")
    s = re.sub(
        r"[^\w\s]",
        " ",
        s
    )
    s = re.sub(
        r"\s+",
        " ",
        s
    )
    return s.strip()
# Canonical keys are used only to generate additional candidates.
# Classification still compares the original normalized fields.
LEGAL = {"inc", "incorporated", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company", "llp"}
ADDRESS_WORDS = {"street":"st", "road":"rd", "avenue":"ave", "boulevard":"blvd", "drive":"dr", "lane":"ln", "apartment":"apt", "suite":"ste", "building":"bldg", "floor":"fl"}

def name_key(s):
    words = norm(s).split()
    return " ".join(w for w in words if w not in LEGAL)

def addr_key(s):
    return " ".join(ADDRESS_WORDS.get(w, w) for w in norm(s).split())

def record(row):
    """
    Convert a source row into a normalized tuple.
    """
    return (
        str(row[0]),
        norm(row[1]),
        norm(row[2]),
        norm(row[3])
    )
def text_of(r):
    """
    Generate retrieval text.
    Business names are repeated to increase their
    influence during vectorization.
    """
    return f"{r[1]} {r[1]} {r[2]}"
# ============================================================
# 2. DATA LOADING
# ============================================================
def source_path(data, split, source):
    return (
        data
        / split
        / f"{split}_source{source}.tsv"
    )
def iter_records(path, chunksize=50000):
    """
    Stream large TSV files in chunks.
    Avoid loading entire source files into pandas.
    """
    reader = pd.read_csv(
        path,
        sep="\t",
        usecols=COLS,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize
    )
    for chunk in reader:
        rows = chunk[COLS].itertuples(
            index=False,
            name=None
        )
        for row in rows:
            yield record(row)
def read_truth(path):
    """
    Load training ground truth.
    Maps Source 1 IDs to sets of matching Source 2
    and Source 3 IDs.
    """
    truth = {}
    with path.open(
        encoding="utf-8",
        newline=""
    ) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t"
        )
        for row in reader:
            source1_id = row[
                "source1_entity_id"
            ]
            matched = row[
                "matched_entity_ids"
            ]
            truth[source1_id] = set(
                filter(
                    None,
                    matched.split(",")
                )
            )
    return truth
def sample_records(path, n, seed=SEED):
    """
    Reservoir sampling.
    Samples records without loading the entire
    source file into memory.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i, r in enumerate(
        iter_records(path)
    ):
        if i < n:
            out.append(r)
        else:
            j = int(
                rng.integers(i + 1)
            )
            if j < n:
                out[j] = r
    return out
# ============================================================
# 3. GPU-ACCELERATED SEMANTIC ENCODER
# ============================================================
class Encoder:
    def __init__(
        self,
        dim,
        device
    ):
        self.dim = dim
        self.device = device
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=2,
            max_features=150000,
            sublinear_tf=True,
            dtype=np.float32
        )
        self.svd = TruncatedSVD(
            n_components=dim,
            n_iter=4,
            random_state=SEED
        )
        self.projection = None
    def fit(
        self,
        paths,
        per_source,
        seed=SEED
    ):
        """
        Fit TF-IDF and SVD on sampled training records.
        SVD fitting runs on the CPU.
        The fitted projection is transferred to CUDA.
        """
        samples = []
        for i, path in enumerate(paths):
            sampled = sample_records(
                path,
                per_source,
                seed + i
            )
            samples.extend(
                text_of(r)
                for r in sampled
            )
        LOG.info(
            "Fitting TF-IDF/SVD on %s sampled records",
            len(samples)
        )
        matrix = (
            self.vectorizer
            .fit_transform(samples)
        )
        self.svd.fit(matrix)
        del matrix
        del samples
        gc.collect()
        self.projection = (
            torch.from_numpy(
                self.svd.components_.T.copy()
            )
            .to(self.device)
        )
    @torch.no_grad()
    def transform(
        self,
        records,
        batch_size=2000
    ):
        """
        Generate normalized dense embeddings.
        Uses CUDA sparse-dense multiplication when
        an NVIDIA GPU is available.
        """
        output = []
        for start in range(
            0,
            len(records),
            batch_size
        ):
            batch = records[
                start:start + batch_size
            ]
            texts = [
                text_of(r)
                for r in batch
            ]
            sparse = (
                self.vectorizer
                .transform(texts)
                .tocoo()
            )
            if self.device.type == "cuda":
                ij = np.vstack(
                    (
                        sparse.row,
                        sparse.col
                    )
                ).astype(np.int64)
                indices = (
                    torch.from_numpy(ij)
                    .to(self.device)
                )
                values = (
                    torch.from_numpy(
                        sparse.data
                    )
                    .to(self.device)
                )
                tensor = (
                    torch.sparse_coo_tensor(
                        indices,
                        values,
                        size=sparse.shape,
                        device=self.device
                    )
                    .coalesce()
                )
                dense = torch.sparse.mm(
                    tensor,
                    self.projection
                ).float()
                dense = (
                    torch.nn.functional
                    .normalize(
                        dense,
                        dim=1
                    )
                )
                arr = dense.cpu().numpy()
                del tensor
                del dense
                del indices
                del values
            else:
                arr = normalize(
                    sparse.tocsr()
                    @ self.svd.components_.T
                ).astype(np.float32)
            output.append(
                np.ascontiguousarray(
                    arr,
                    dtype=np.float32
                )
            )
        if not output:
            return np.empty(
                (0, self.dim),
                dtype=np.float32
            )
        return np.concatenate(output)
# ============================================================
# 4. FAISS TARGET INDEX
# ============================================================
class TargetStore:
    """
    Disk-backed target metadata and FAISS HNSW indexes.
    Separate indexes are constructed for:
        Source 2 + country
        Source 3 + country
    Country labels are discovered dynamically.
    SQLite stores record metadata.
    FAISS indexes remain in system RAM.
    """
    def __init__(
        self,
        db_path,
        dim,
        m=16,
        ef_construction=100,
        ef_search=192,
        cache_dir=None,
        rebuild=False
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.rebuild = rebuild
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(
            "PRAGMA journal_mode=WAL"
        )
        self.conn.execute(
            "PRAGMA synchronous=OFF"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS targets (
                id TEXT PRIMARY KEY,
                name TEXT,
                addr TEXT,
                country TEXT,
                source INTEGER,
                name_key TEXT,
                addr_key TEXT
            )
            """
        )

        self.indexes = {}
        self.ids = {}
        self.dim = dim
        self.m = m
        self.ef_construction = (
            ef_construction
        )
        self.ef_search = ef_search
    def build(
        self,
        data,
        split,
        encoder,
        chunk_size=20000
    ):
        """
        Build FAISS indexes from Source 2 and Source 3.
        Embeddings are generated in batches.
        """
        if self.cache_dir and not self.rebuild and Path(self.conn.execute("PRAGMA database_list").fetchone()[2]).exists() and (self.cache_dir / "complete.joblib").exists():
            meta = joblib.load(self.cache_dir / "complete.joblib")
            if meta.get("dim") == self.dim and meta.get("m") == self.m:
                LOG.info("Loading cached %s FAISS indexes", split)
                for source, country in meta["keys"]:
                    tag = f"{source}_{country}"
                    ix = faiss.read_index(str(self.cache_dir / f"{tag}.faiss"))
                    ix.hnsw.efSearch = self.ef_search
                    self.indexes[(source, country)] = ix
                    self.ids[(source, country)] = joblib.load(self.cache_dir / f"{tag}.ids.joblib")
                return
        if self.cache_dir:
            (self.cache_dir / "complete.joblib").unlink(missing_ok=True)
        self.conn.execute("DROP INDEX IF EXISTS ix_target_name")
        self.conn.execute("DROP INDEX IF EXISTS ix_target_addr")
        self.conn.execute("DELETE FROM targets")
        self.conn.commit()

        counts = defaultdict(int)
        # Count records per source and country.
        for source in (2, 3):
            path = source_path(
                data,
                split,
                source
            )
            reader = pd.read_csv(
                path,
                sep="\t",
                usecols=["country"],
                dtype=str,
                keep_default_na=False,
                chunksize=100000
            )
            for chunk in reader:
                values = (
                    chunk["country"]
                    .map(norm)
                    .value_counts()
                )
                for country, count in values.items():
                    counts[
                        (source, country)
                    ] += int(count)
        # Create FAISS indexes.
        for key, count in counts.items():
            index = faiss.IndexHNSWFlat(
                self.dim,
                self.m,
                faiss.METRIC_INNER_PRODUCT
            )
            index.hnsw.efConstruction = (
                self.ef_construction
            )
            index.hnsw.efSearch = (
                self.ef_search
            )
            self.indexes[key] = index
            self.ids[key] = []
            LOG.info(
                "Index %s: %s records",
                key,
                count
            )
        # Populate indexes.
        for source in (2, 3):
            path = source_path(
                data,
                split,
                source
            )
            buf = []
            processed = 0
            for r in iter_records(path):
                buf.append(r)
                if len(buf) >= chunk_size:
                    self._add(
                        buf,
                        source,
                        encoder
                    )
                    processed += len(buf)
                    if (
                        processed % 200000
                        < chunk_size
                    ):
                        LOG.info(
                            "Source %s indexed %s",
                            source,
                            processed
                        )
                    buf.clear()
            if buf:
                self._add(
                    buf,
                    source,
                    encoder
                )
            self.conn.commit()

        LOG.info("Building SQLite exact-match indexes")
        self.conn.execute("CREATE INDEX ix_target_name ON targets(country, name_key)")
        self.conn.execute("CREATE INDEX ix_target_addr ON targets(country, addr_key)")
        self.conn.commit()
        if self.cache_dir:
            for (source, country), ix in self.indexes.items():
                tag = f"{source}_{country}"
                faiss.write_index(ix, str(self.cache_dir / f"{tag}.faiss"))
                joblib.dump(self.ids[(source, country)], self.cache_dir / f"{tag}.ids.joblib", compress=0)
            joblib.dump({"dim":self.dim, "m":self.m, "keys":list(self.indexes)},
                        self.cache_dir / "complete.joblib")
            LOG.info("Saved reusable FAISS indexes and metadata")

    def _add(
        self,
        records,
        source,
        encoder
    ):
        """
        Generate embeddings and add them to FAISS.
        """
        vectors = encoder.transform(
            records
        )
        groups = defaultdict(list)
        for i, r in enumerate(records):
            groups[r[3]].append(i)
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO targets
                (id,name,addr,country,source,name_key,addr_key)
                VALUES (?,?,?,?,?,?,?)
            """,
            [(r[0],r[1],r[2],r[3],source,name_key(r[1]),addr_key(r[2])) for r in records]
        )
        for country, positions in groups.items():
            key = (
                source,
                country
            )
            v = np.ascontiguousarray(
                vectors[positions],
                dtype=np.float32
            )
            self.indexes[key].add(v)
            self.ids[key].extend(
                records[i][0]
                for i in positions
            )
        self.conn.commit()
    def get_many(self, ids):
        """
        Retrieve target metadata in SQLite batches.
        """
        out = {}
        ids = list(
            dict.fromkeys(ids)
        )
        for start in range(
            0,
            len(ids),
            400
        ):
            part = ids[
                start:start + 400
            ]
            if not part:
                continue
            placeholders = ",".join(
                "?" * len(part)
            )
            query = (
                "SELECT id,name,addr,country "
                "FROM targets WHERE id IN ("
                + placeholders
                + ")"
            )
            for r in self.conn.execute(
                query,
                part
            ):
                out[r[0]] = r
        return out
    def exact_candidates(self, records, cap=30):
        """Batch SQL lookups for exact canonical name and address matches.
        Common/generic keys are capped to avoid candidate explosions.
        """
        result = [[] for _ in records]
        for field, fn, minlen in (("name_key", name_key, 4), ("addr_key", addr_key, 8)):
            field_cap = cap if field == "name_key" else max(10, cap // 2)
            groups = defaultdict(list)
            for i, r in enumerate(records):
                key = fn(r[1] if field == "name_key" else r[2])
                if len(key) >= minlen:
                    groups[(r[3], key)].append(i)
            by_country = defaultdict(list)
            for country, key in groups:
                by_country[country].append(key)
            for country, keys in by_country.items():
                for start in range(0, len(keys), 250):
                    part = keys[start:start+250]
                    placeholders = ",".join("?" for _ in part)
                    sql = f"SELECT id,{field} FROM targets WHERE country=? AND {field} IN ({placeholders})"
                    hits = defaultdict(list)
                    for cid, key in self.conn.execute(sql, [country] + part):
                        if len(hits[key]) < field_cap:
                            hits[key].append(cid)
                    for key in part:
                        for i in groups[(country,key)]:
                            for cid in hits.get(key, [])[:field_cap]:
                                result[i].append((cid, 0.0))
        return result

    def retrieve(
        self,
        records,
        encoder,
        top_k
    ):
        """
        Retrieve nearest neighbors for each Source 1
        record from both target sources.
        """
        vectors = encoder.transform(
            records
        )
        output = [
            []
            for _ in records
        ]
        grouped = defaultdict(list)
        for i, r in enumerate(records):
            grouped[r[3]].append(i)
        for country, positions in grouped.items():
            q = np.ascontiguousarray(
                vectors[positions],
                dtype=np.float32
            )
            for source in (2, 3):
                key = (
                    source,
                    country
                )
                if key not in self.indexes:
                    continue
                index = self.indexes[key]
                if index.ntotal == 0:
                    continue
                count = min(
                    top_k,
                    index.ntotal
                )
                scores, labels = (
                    index.search(
                        q,
                        count
                    )
                )
                ids = self.ids[key]
                for local_i, original_i in enumerate(
                    positions
                ):
                    matches = []
                    for label, score in zip(
                        labels[local_i],
                        scores[local_i]
                    ):
                        if label < 0:
                            continue
                        matches.append(
                            (
                                ids[int(label)],
                                float(score)
                            )
                        )
                    output[
                        original_i
                    ].extend(matches)
        exact = self.exact_candidates(records)
        for i, extra in enumerate(exact):
            existing = {cid for cid, _ in output[i]}
            for cid, _ in extra:
                if cid not in existing:
                    output[i].append((cid, -1.0))
                    existing.add(cid)
        return output
    def close(self):
        self.conn.close()
# ============================================================
# 5. SIMILARITY FEATURE ENGINEERING
# ============================================================
def jaccard(a, b):
    a = set(a.split())
    b = set(b.split())
    if not a or not b:
        return 0.0
    return (
        len(a & b)
        / len(a | b)
    )
def features(
    a,
    b,
    similarity
):
    """
    Extract numerical similarity features.
    Includes:
        Semantic similarity
        Fuzzy name similarity
        Fuzzy address similarity
        Token overlap
        Address number agreement
        Missing-value indicators
        Country agreement
    """
    an, aa, ac = a[1:]
    bn, ba, bc = b[1:]
    na = set(
        re.findall(
            r"\d+",
            aa
        )
    )
    nb = set(
        re.findall(
            r"\d+",
            ba
        )
    )
    return [
        similarity,
        fuzz.ratio(an, bn) / 100,
        fuzz.token_sort_ratio(
            an,
            bn
        ) / 100,
        fuzz.token_set_ratio(
            an,
            bn
        ) / 100,
        fuzz.WRatio(
            an,
            bn
        ) / 100,
        fuzz.ratio(
            aa,
            ba
        ) / 100,
        fuzz.token_sort_ratio(
            aa,
            ba
        ) / 100,
        fuzz.token_set_ratio(
            aa,
            ba
        ) / 100,
        jaccard(
            an,
            bn
        ),
        jaccard(
            aa,
            ba
        ),
        float(
            bool(an)
            and an == bn
        ),
        float(
            bool(aa)
            and aa == ba
        ),
        float(
            bool(na & nb)
        ),
        float(
            bool(na)
            and bool(nb)
        ),
        float(
            bool(na)
            and bool(nb)
            and na != nb
        ),
        abs(
            len(an) - len(bn)
        ),
        abs(
            len(aa) - len(ba)
        ),
        float(not an),
        float(not bn),
        float(not aa),
        float(not ba),
        float(ac == bc),
        float(bool(name_key(an)) and name_key(an) == name_key(bn)),
        float(bool(addr_key(aa)) and addr_key(aa) == addr_key(ba)),
        float(bool(an and bn) and (an in bn or bn in an)),
        float(bool(aa and ba) and (aa in ba or ba in aa))
    ]
# ============================================================
# 6. CANDIDATE RECALL
# ============================================================
def candidate_recall(
    records,
    retrieved,
    truth
):
    """
    Calculate recall of FAISS candidate retrieval.
    """
    total = 0
    found = 0
    for r, candidates in zip(
        records,
        retrieved
    ):
        gt = truth.get(
            r[0],
            set()
        )
        predicted = {
            cid
            for cid, _ in candidates
        }
        total += len(gt)
        found += len(
            gt & predicted
        )
    score = (
        found / total
        if total
        else 0.0
    )
    LOG.info(
        "Candidate recall: %.5f (%s/%s)",
        score,
        found,
        total
    )
    return score
# ============================================================
# 7. TRAINING PAIR GENERATION
# ============================================================
def training_arrays(
    records,
    candidates,
    truth,
    store,
    max_neg=8
):
    """
    Build supervised training pairs.
    Positives:
        Only known matches that candidate generation actually retrieved.
    Hard negatives:
        Similar retrieved records not listed as matches.
    """
    X = []
    y = []
    for r, retrieved in zip(
        records,
        candidates
    ):
        actual = truth.get(
            r[0],
            set()
        )
        scores = dict(
            retrieved
        )
        sorted_candidates = sorted(
            retrieved,
            key=lambda p: -p[1]
        )
        hard = [cid for cid, sim in sorted_candidates if cid not in actual and sim == -1.0]
        ann = [cid for cid, sim in sorted_candidates if cid not in actual and sim != -1.0]
        negative = list(dict.fromkeys(hard[:max_neg // 2] + ann[:max_neg]))[:max_neg]
        selected = (
            [cid for cid, _ in sorted_candidates if cid in actual]
            + negative
        )
        targets = store.get_many(
            selected
        )
        for cid in selected:
            if cid not in targets:
                continue
            # All training pairs now come from the real candidate generator.
            similarity = scores.get(
                cid,
                0.0
            )
            X.append(
                features(
                    r,
                    targets[cid],
                    similarity
                )
            )
            y.append(
                int(cid in actual)
            )
    if not X or len(set(y)) != 2:
        raise RuntimeError(
            "Training requires both positive "
            "and negative examples."
        )
    return (
        np.asarray(
            X,
            dtype=np.float32
        ),
        np.asarray(
            y,
            dtype=np.int8
        )
    )
# ============================================================
# 8. GPU XGBOOST CLASSIFIER
# ============================================================
def fit_model(
    X,
    y,
    device,
    cpu_threads=8
):
    """
    Train XGBoost using GPU histogram training
    when CUDA is available.
    """
    model = xgb.XGBClassifier(
        n_estimators=450,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.9,
        tree_method="hist",
        device=device,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=cpu_threads
    )
    LOG.info(
        "Training XGBoost on %s",
        device
    )
    model.fit(
        X,
        y,
        verbose=False
    )
    return model
# ============================================================
# 9. BATCHED INFERENCE
# ============================================================
def score_batch(
    records,
    candidates,
    store,
    model
):
    """
    Generate features and predict match probabilities
    for one batch of reference entities.
    """
    X = []
    meta = []
    all_ids = [
        cid
        for row in candidates
        for cid, _ in row
    ]
    targets = store.get_many(
        all_ids
    )
    out = {
        r[0]: []
        for r in records
    }
    for r, row in zip(
        records,
        candidates
    ):
        for cid, sim in row:
            if cid not in targets:
                continue
            X.append(
                features(
                    r,
                    targets[cid],
                    sim
                )
            )
            meta.append(
                (
                    r[0],
                    cid
                )
            )
    if X:
        matrix = np.asarray(
            X,
            dtype=np.float32
        )
        probs = model.predict_proba(
            matrix
        )[:, 1]
        for (rid, cid), p in zip(
            meta,
            probs
        ):
            out[rid].append(
                (
                    cid,
                    float(p)
                )
            )
    return out
# ============================================================
# 10. OFFICIAL MACRO F0.5 EVALUATION
# ============================================================
def entity_f05(
    actual,
    predicted
):
    """
    Compute F0.5 for one Source 1 entity.
    Correctly predicted singletons receive 1.0.
    """
    if not actual and not predicted:
        return 1.0
    if not actual or not predicted:
        return 0.0
    tp = len(
        actual & predicted
    )
    if not tp:
        return 0.0
    return (
        1.25 * tp
        / (
            0.25 * len(actual)
            + len(predicted)
        )
    )
# ============================================================
# 11. THRESHOLD OPTIMIZATION
# ============================================================
def choose_threshold(
    scored,
    truth
):
    """
    Optimize classification threshold using
    validation macro F0.5.
    """
    best_score = float("-inf")
    best_threshold = 0.75
    thresholds = np.arange(
        0.30,
        0.991,
        0.025
    )
    for threshold in thresholds:
        scores = []
        for rid, pairs in scored.items():
            predicted = {
                cid
                for cid, p in pairs
                if p >= threshold
            }
            actual = truth.get(
                rid,
                set()
            )
            scores.append(
                entity_f05(
                    actual,
                    predicted
                )
            )
        score = float(
            np.mean(scores)
        )
        LOG.info(
            "Threshold %.3f | Macro F0.5 %.5f",
            threshold,
            score
        )
        if score > best_score:
            best_score = score
            best_threshold = float(
                threshold
            )
    LOG.info(
        "Best validation F0.5: %.5f",
        best_score
    )
    LOG.info(
        "Selected threshold: %.3f",
        best_threshold
    )
    return best_threshold
# ============================================================
# 12. TEST INFERENCE AND SUBMISSION GENERATION
# ============================================================
def write_test(
    data,
    encoder,
    model,
    threshold,
    args
):
    """
    Run inference over the complete test dataset.
    Writes:
        matching_results.tsv
        candidate_pairs.tsv
    """
    LOG.info(
        "Building TEST target indexes."
    )
    db = (
        args.work
        / f"test_targets_d{args.dim}_s{args.svd_samples}.sqlite"
    )
    if db.exists() and args.rebuild:
        db.unlink()
    store = TargetStore(
        db,
        args.dim,
        ef_search=args.ef_search,
        cache_dir=args.work / f"test_indexes_d{args.dim}_s{args.svd_samples}",
        rebuild=args.rebuild
    )
    store.build(
        data,
        "test",
        encoder,
        chunk_size=args.index_batch
    )
    match_path = (
        args.output
        / "matching_results.tsv"
    )
    cand_path = (
        args.output
        / "candidate_pairs.tsv"
    )
    with (
        match_path.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as mf,
        cand_path.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as cf
    ):
        mw = csv.writer(
            mf,
            delimiter="\t",
            lineterminator="\n"
        )
        cw = csv.writer(
            cf,
            delimiter="\t",
            lineterminator="\n"
        )
        mw.writerow([
            "source1_entity_id",
            "matched_entity_ids"
        ])
        cw.writerow([
            "source1_entity_id",
            "candidate_entity_ids"
        ])
        batch = []
        count = 0
        def process(records):
            """
            Retrieve, classify and write one batch.
            """
            retrieved = store.retrieve(
                records,
                encoder,
                args.top_k
            )
            scored = score_batch(
                records,
                retrieved,
                store,
                model
            )
            for r, pairs in zip(
                records,
                retrieved
            ):
                rid = r[0]
                cids = list(
                    dict.fromkeys(
                        cid
                        for cid, _ in pairs
                    )
                )
                matched = list(
                    dict.fromkeys(
                        cid
                        for cid, p in scored[rid]
                        if p >= threshold
                    )
                )
                cw.writerow([
                    rid,
                    ",".join(cids)
                ])
                mw.writerow([
                    rid,
                    ",".join(matched)
                ])
        test_path = source_path(
            data,
            "test",
            1
        )
        for r in iter_records(
            test_path
        ):
            batch.append(r)
            if len(batch) >= args.query_batch:
                process(batch)
                count += len(batch)
                LOG.info(
                    "Test S1 processed: %s",
                    count
                )
                batch.clear()
        if batch:
            process(batch)
            count += len(batch)
        LOG.info(
            "Wrote %s test S1 rows",
            count
        )
    store.close()
    # --------------------------------------------------------
    # OFFICIAL VALIDATION
    # --------------------------------------------------------
    validator = (
        data.parent
        / "utils"
        / "validate_submission.py"
    )
    if not validator.exists():
        validator = (
            data.parent
            / "validate_submission.py"
        )
    if validator.exists():
        LOG.info(
            "Running official validator."
        )
        subprocess.run(
            [
                sys.executable,
                str(validator),
                "--matching",
                str(match_path),
                "--candidate",
                str(cand_path),
                "--test-dir",
                str(data / "test")
            ],
            check=True
        )
    else:
        LOG.warning(
            "Validator not found. "
            "Run the official validator manually."
        )
# ============================================================
# 13. MAIN PIPELINE
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "GPU-accelerated Amazon ML "
            "Business Entity Resolution"
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=30000
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64
    )
    parser.add_argument(
        "--svd-samples",
        type=int,
        default=60000
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=min(12, os.cpu_count() or 8),
        help="CPU threads for FAISS, PyTorch CPU ops and XGBoost; benchmark rather than maximizing blindly"
    )
    parser.add_argument(
        "--query-batch",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--index-batch",
        type=int,
        default=10000
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=192
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help=(
            "Train and validate only. "
            "Skip full test inference."
        )
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Disable GPU acceleration."
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild encoder and indexes")

    args = parser.parse_args()
    faiss.omp_set_num_threads(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(message)s"
        )
    )
    # --------------------------------------------------------
    # DIRECTORY CONFIGURATION
    # --------------------------------------------------------
    base = (
        Path(__file__)
        .resolve()
        .parent
    )
    if args.data is None:
        if (
            base / "dataset"
        ).exists():
            args.data = (
                base / "dataset"
            )
        else:
            args.data = (
                base
                / "student_resource"
                / "dataset"
            )
    args.output = (
        args.data.parent
        / "output"
    )
    args.work = (
        args.data.parent
        / "artifacts_gpu"
    )
    args.output.mkdir(
        parents=True,
        exist_ok=True
    )
    args.work.mkdir(
        parents=True,
        exist_ok=True
    )
    # --------------------------------------------------------
    # GPU DETECTION
    # --------------------------------------------------------
    cuda_available = (
        torch.cuda.is_available()
    )
    if args.cpu or not cuda_available:
        device = torch.device(
            "cpu"
        )
    else:
        device = torch.device(
            "cuda"
        )
    LOG.info(
        "PyTorch device: %s",
        device
    )
    LOG.info(
        "CUDA available: %s",
        cuda_available
    )
    if device.type == "cuda":
        gpu_name = (
            torch.cuda
            .get_device_name(0)
        )
        gpu_memory = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
            / 2**30
        )
        LOG.info(
            "GPU: %s",
            gpu_name
        )
        LOG.info(
            "VRAM: %.2f GiB",
            gpu_memory
        )
    else:
        LOG.warning(
            "CUDA acceleration disabled. "
            "Running on CPU."
        )
    # --------------------------------------------------------
    # TRAINING PATHS
    # --------------------------------------------------------
    paths = [
        source_path(
            args.data,
            "train",
            s
        )
        for s in (1, 2, 3)
    ]
    # --------------------------------------------------------
    # SEMANTIC ENCODER
    # --------------------------------------------------------
    LOG.info(
        "Training semantic encoder."
    )
    encoder = Encoder(
        args.dim,
        device
    )
    enc_cache = args.work / f"encoder_d{args.dim}_s{args.svd_samples}.joblib"
    if enc_cache.exists() and not args.rebuild:
        LOG.info("Loading cached encoder")
        encoder = joblib.load(enc_cache)
        encoder.device = device
        encoder.projection = torch.from_numpy(encoder.svd.components_.T.copy()).to(device)
    else:
        encoder.fit(paths, args.svd_samples)
    # Temporarily remove the CUDA tensor before
    # serializing the encoder.
    projection = (
        encoder.projection
    )
    encoder.projection = None
    joblib.dump(
        encoder,
        args.work
        / enc_cache.name
    )
    encoder.projection = (
        projection
    )
    # --------------------------------------------------------
    # LOAD GROUND TRUTH
    # --------------------------------------------------------
    truth = read_truth(
        args.data
        / "train"
        / "train_ground_truth.tsv"
    )
    # --------------------------------------------------------
    # SAMPLE TRAINING REFERENCES
    # --------------------------------------------------------
    LOG.info(
        "Sampling %s training references.",
        args.sample
    )
    sampled = sample_records(
        paths[0],
        args.sample
    )
    train, valid = train_test_split(
        sampled,
        test_size=0.2,
        random_state=SEED
    )
    # --------------------------------------------------------
    # BUILD TRAINING INDEXES
    # --------------------------------------------------------
    db = (
        args.work
        / f"train_targets_d{args.dim}_s{args.svd_samples}.sqlite"
    )
    if db.exists() and args.rebuild:
        db.unlink()
    store = TargetStore(
        db,
        args.dim,
        ef_search=args.ef_search,
        cache_dir=args.work / f"train_indexes_d{args.dim}_s{args.svd_samples}",
        rebuild=args.rebuild
    )
    LOG.info(
        "Building training FAISS indexes."
    )
    store.build(
        args.data,
        "train",
        encoder,
        chunk_size=args.index_batch
    )
    # --------------------------------------------------------
    # TRAINING CANDIDATES
    # --------------------------------------------------------
    LOG.info(
        "Retrieving training candidates."
    )
    train_candidates = (
        store.retrieve(
            train,
            encoder,
            args.top_k
        )
    )
    candidate_recall(
        train,
        train_candidates,
        truth
    )
    # --------------------------------------------------------
    # TRAINING FEATURES
    # --------------------------------------------------------
    LOG.info(
        "Generating training features."
    )
    X, y = training_arrays(
        train,
        train_candidates,
        truth,
        store
    )
    LOG.info(
        "Training pairs: %s",
        len(y)
    )
    LOG.info(
        "Positive pairs: %s",
        int(y.sum())
    )
    # --------------------------------------------------------
    # GPU CLASSIFIER TRAINING
    # --------------------------------------------------------
    xgb_device = (
        "cuda"
        if device.type == "cuda"
        else "cpu"
    )
    model = fit_model(
        X,
        y,
        xgb_device,
        args.cpu_threads
    )
    model.save_model(
        str(
            args.work
            / "xgboost.json"
        )
    )
    del X
    del y
    del train_candidates
    gc.collect()
    # --------------------------------------------------------
    # VALIDATION RETRIEVAL
    # --------------------------------------------------------
    LOG.info(
        "Retrieving validation candidates."
    )
    valid_candidates = (
        store.retrieve(
            valid,
            encoder,
            args.top_k
        )
    )
    candidate_recall(
        valid,
        valid_candidates,
        truth
    )
    # --------------------------------------------------------
    # VALIDATION INFERENCE
    # --------------------------------------------------------
    scored = {}
    for start in range(
        0,
        len(valid),
        args.query_batch
    ):
        batch_records = valid[
            start:start + args.query_batch
        ]
        batch_candidates = valid_candidates[
            start:start + args.query_batch
        ]
        batch_scores = score_batch(
            batch_records,
            batch_candidates,
            store,
            model
        )
        scored.update(
            batch_scores
        )
    # --------------------------------------------------------
    # THRESHOLD OPTIMIZATION
    # --------------------------------------------------------
    threshold = choose_threshold(
        scored,
        truth
    )
    joblib.dump(
        {
            "threshold": threshold
        },
        args.work
        / "threshold.joblib"
    )
    # --------------------------------------------------------
    # MEMORY CLEANUP
    # --------------------------------------------------------
    del scored
    del valid_candidates
    del train
    del valid
    del truth
    del sampled
    store.close()
    del store
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # --------------------------------------------------------
    # TEST INFERENCE
    # --------------------------------------------------------
    if not args.skip_test:
        write_test(
            args.data,
            encoder,
            model,
            threshold,
            args
        )
    LOG.info(
        "Pipeline completed successfully."
    )
# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
