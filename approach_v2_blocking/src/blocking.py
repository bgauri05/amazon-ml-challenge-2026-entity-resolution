"""
Approach V2 Multi-Pass Candidate Generator (Baseline, Tuned, Relevance-Ranked, Numeric-Normalized)
Queries SQLite inverted index to generate candidate pairs for S1 entities.
Performs union deduplication, tracks candidate rules, relevance pre-ranking, and enforces MAX_CANDIDATES_PER_ENTITY.
"""

import csv
import re
import sys
import time
import sqlite3
import argparse
import polars as pl
import numpy as np
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(2147483647)

# Add approach root to sys.path
SRC_DIR = Path(__file__).resolve().parent
APPROACH_ROOT = SRC_DIR.parent
if str(APPROACH_ROOT) not in sys.path:
    sys.path.insert(0, str(APPROACH_ROOT))

from src.config import (
    TRAIN_FILES,
    INDEX_DB_PATH,
    INDEX_TUNED_DB_PATH,
    CANDIDATE_FILES,
    MAX_CANDIDATES_PER_ENTITY,
    MAX_CANDIDATES_PER_ENTITY_TUNED,
    RELEVANCE_WEIGHTS
)
from src.build_indexes import (
    fast_normalize_name,
    fast_normalize_address,
    fast_normalize_country,
    extract_blocking_keys_fast,
    extract_blocking_keys_fast_tuned,
    build_inverted_indexes
)

RE_POSTAL = re.compile(r"\b\d{5,6}\b")


def compute_candidate_relevance_score(
    s1_name_norm: str,
    s1_addr_norm: str,
    cand_name_norm: str,
    cand_addr_norm: str,
    passes_set: set
) -> float:
    """
    Computes a lightweight, deterministic candidate relevance score using normalized text fields.
    """
    w = RELEVANCE_WEIGHTS

    # 1. Name Token Jaccard Similarity
    s1_ntoks = set(s1_name_norm.split()) if s1_name_norm else set()
    cand_ntoks = set(cand_name_norm.split()) if cand_name_norm else set()
    s1_n_meaningful = {t for t in s1_ntoks if len(t) >= 2}
    cand_n_meaningful = {t for t in cand_ntoks if len(t) >= 2}

    if s1_n_meaningful and cand_n_meaningful:
        name_jaccard = len(s1_n_meaningful & cand_n_meaningful) / len(s1_n_meaningful | cand_n_meaningful)
    else:
        name_jaccard = 0.0

    # 2. Address Token Jaccard Similarity
    s1_atoks = set(s1_addr_norm.split()) if s1_addr_norm else set()
    cand_atoks = set(cand_addr_norm.split()) if cand_addr_norm else set()
    s1_a_meaningful = {t for t in s1_atoks if len(t) >= 2}
    cand_a_meaningful = {t for t in cand_atoks if len(t) >= 2}

    if s1_a_meaningful and cand_a_meaningful:
        address_jaccard = len(s1_a_meaningful & cand_a_meaningful) / len(s1_a_meaningful | cand_a_meaningful)
    else:
        address_jaccard = 0.0

    # 3. Exact Name Bonus
    exact_name_bonus = w["exact_name_bonus"] if (s1_name_norm and s1_name_norm == cand_name_norm) else 0.0

    # 4. Postal Code Match Bonus
    s1_postals = set(RE_POSTAL.findall(s1_addr_norm)) if s1_addr_norm else set()
    cand_postals = set(RE_POSTAL.findall(cand_addr_norm)) if cand_addr_norm else set()
    postal_match_bonus = w["postal_match_bonus"] if (s1_postals and cand_postals and (s1_postals & cand_postals)) else 0.0

    # 5. Numeric Overlap Bonus
    s1_nums = {t for t in s1_atoks if t.isdigit()}
    cand_nums = {t for t in cand_atoks if t.isdigit()}
    numeric_overlap_bonus = w["numeric_overlap_bonus"] if (s1_nums and cand_nums and (s1_nums & cand_nums)) else 0.0

    # 6. Multi-Pass Bonus
    multi_pass_bonus = w["multi_pass_bonus"] * len(passes_set)

    score = (
        name_jaccard * w["name_jaccard_weight"] +
        address_jaccard * w["address_jaccard_weight"] +
        exact_name_bonus +
        postal_match_bonus +
        numeric_overlap_bonus +
        multi_pass_bonus
    )

    return round(score, 4)


def load_candidate_text_cache(target_cand_ids: set) -> dict:
    """
    Streams S2 and S3 TSVs to populate normalized text attributes for target candidate IDs.
    """
    print(f"Loading text attributes for {len(target_cand_ids):,} unique candidate IDs...", flush=True)
    t0 = time.time()
    cache = {}

    s2_ids = {cid for cid in target_cand_ids if cid.startswith("S2-")}
    s3_ids = {cid for cid in target_cand_ids if cid.startswith("S3-")}

    if s2_ids:
        with open(TRAIN_FILES["s2"], mode="r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                cid = row.get("entity_id", "").strip()
                if cid in s2_ids:
                    cache[cid] = (
                        fast_normalize_name(row.get("business_name")),
                        fast_normalize_address(row.get("business_address"))
                    )

    if s3_ids:
        with open(TRAIN_FILES["s3"], mode="r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                cid = row.get("entity_id", "").strip()
                if cid in s3_ids:
                    cache[cid] = (
                        fast_normalize_name(row.get("business_name")),
                        fast_normalize_address(row.get("business_address"))
                    )

    print(f"  [DONE] Cached text for {len(cache):,} candidate IDs in {round(time.time() - t0, 2)}s", flush=True)
    return cache


def run_blocking_test_1k(
    s1_filepath: Path,
    output_parquet: Path,
    num_entities: int = 1000,
    tuned: bool = False,
    ranked: bool = False
):
    label_mode = "TUNED + RELEVANCE RANKING (NUMERIC NORMALIZED)" if (tuned and ranked) else ("TUNED (UNRANKED)" if tuned else "BASELINE")
    db_path = INDEX_TUNED_DB_PATH if tuned else INDEX_DB_PATH
    max_cands = MAX_CANDIDATES_PER_ENTITY_TUNED if tuned else MAX_CANDIDATES_PER_ENTITY

    print("=" * 80)
    print(f"APPROACH V2: CANDIDATE GENERATION TEST ({label_mode}, FIRST {num_entities:,} S1 ENTITIES)")
    print("=" * 80)
    t0 = time.time()

    if not db_path.exists():
        print(f"⚠️ Index database not found at {db_path.name}. Building index first...")
        build_inverted_indexes(db_path=db_path, tuned=tuned)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    key_extractor = extract_blocking_keys_fast_tuned if tuned else extract_blocking_keys_fast

    # Load 1,000 S1 entity rows
    s1_rows = []
    with open(s1_filepath, mode="r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_rows.append({
                "s1_id": row.get("entity_id", "").strip(),
                "name_norm": fast_normalize_name(row.get("business_name")),
                "addr_norm": fast_normalize_address(row.get("business_address")),
                "country_norm": fast_normalize_country(row.get("country"))
            })
            if len(s1_rows) >= num_entities:
                break

    # Phase 1: Retrieve un-capped raw candidates from SQLite index for all 1,000 S1 entities
    print(f"\nRetrieving candidate keys from {db_path.name}...", flush=True)
    s1_raw_cands = {}
    all_cand_ids = set()

    for item in s1_rows:
        s1_id = item["s1_id"]
        keys = key_extractor(item["name_norm"], item["addr_norm"], item["country_norm"])

        cand_passes = defaultdict(set)
        for pass_id, country, block_key in keys:
            cur.execute(
                "SELECT entity_id FROM inverted_index WHERE pass_id = ? AND country = ? AND block_key = ?;",
                (pass_id, country, block_key)
            )
            for (cand_id,) in cur.fetchall():
                cand_passes[cand_id].add(pass_id)
                all_cand_ids.add(cand_id)

        s1_raw_cands[s1_id] = cand_passes

    conn.close()

    # Phase 2: If ranked, load candidate text cache and compute relevance score
    cand_text_cache = {}
    if ranked:
        cand_text_cache = load_candidate_text_cache(all_cand_ids)

    # Phase 3: Rank & Cap Candidates per S1 entity
    print("\nRanking & capping candidates per S1 entity...", flush=True)
    all_candidates = []
    cands_per_s1 = []
    zero_cand_s1_count = 0
    rule_counts = defaultdict(int)

    for item in s1_rows:
        s1_id = item["s1_id"]
        cand_passes = s1_raw_cands[s1_id]

        if not cand_passes:
            cands_per_s1.append(0)
            zero_cand_s1_count += 1
            continue

        if ranked:
            # Relevance-based sorting
            scored_cands = []
            s1_n = item["name_norm"]
            s1_a = item["addr_norm"]

            for cand_id, passes_set in cand_passes.items():
                cand_n, cand_a = cand_text_cache.get(cand_id, ("", ""))
                score = compute_candidate_relevance_score(s1_n, s1_a, cand_n, cand_a, passes_set)
                scored_cands.append((cand_id, passes_set, score))

            # Sort by (-score, candidate_id)
            ranked_cands = sorted(scored_cands, key=lambda x: (-x[2], x[0]))
            top_cands = ranked_cands[:max_cands]
        else:
            # Baseline sorting by (-len(passes), candidate_id)
            ranked_cands = sorted(cand_passes.items(), key=lambda x: (-len(x[1]), x[0]))
            top_cands = [(cid, passes, 0.0) for cid, passes in ranked_cands[:max_cands]]

        cands_per_s1.append(len(top_cands))

        for cand_tuple in top_cands:
            cand_id = cand_tuple[0]
            passes_set = cand_tuple[1]
            cand_src = "S2" if cand_id.startswith("S2-") else ("S3" if cand_id.startswith("S3-") else "UNKNOWN")
            passes_str = ",".join(sorted(passes_set))

            all_candidates.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "candidate_source": cand_src,
                "blocking_passes": passes_str
            })

            for p in passes_set:
                rule_counts[p] += 1

    t_elapsed = round(time.time() - t0, 2)

    # Compute Statistics
    counts_arr = np.array(cands_per_s1)
    avg_cands = round(float(np.mean(counts_arr)), 2) if len(counts_arr) > 0 else 0.0
    median_cands = float(np.median(counts_arr)) if len(counts_arr) > 0 else 0.0
    p95_cands = float(np.percentile(counts_arr, 95)) if len(counts_arr) > 0 else 0.0
    max_cands_obs = int(np.max(counts_arr)) if len(counts_arr) > 0 else 0

    df_cands = pl.DataFrame(all_candidates) if all_candidates else pl.DataFrame({
        "source1_entity_id": [],
        "candidate_entity_id": [],
        "candidate_source": [],
        "blocking_passes": []
    })

    unique_candidate_pairs = len(df_cands)
    df_cands.write_parquet(output_parquet, compression="snappy")

    print(f"\n[SUMMARY FOR FIRST {len(s1_rows):,} S1 ENTITIES ({label_mode})]")
    print("-" * 60)
    print(f"Total S1 Entities Processed     : {len(s1_rows):,}")
    print(f"Total Unique Candidate Pairs     : {unique_candidate_pairs:,}")
    print(f"S1 Entities with 0 Candidates   : {zero_cand_s1_count:,} ({round(zero_cand_s1_count/len(s1_rows)*100, 2)}%)")
    print("-" * 60)
    print("CANDIDATE DENSITY METRICS:")
    print(f"  - Average Candidates per S1   : {avg_cands}")
    print(f"  - Median Candidates per S1    : {median_cands}")
    print(f"  - P95 Candidates per S1       : {p95_cands}")
    print(f"  - Maximum Candidates per S1   : {max_cands_obs}")
    print("-" * 60)
    print("CANDIDATE CONTRIBUTIONS BY BLOCKING RULE:")
    for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"  - Rule [{rule:15s}] : {count:,} candidate links")
    print("-" * 60)
    print(f"Saved test candidates to        : {output_parquet.name}")
    print(f"Execution Time                  : {t_elapsed}s")
    print("=" * 80)

    return {
        "processed": len(s1_rows),
        "unique_candidates": unique_candidate_pairs,
        "zero_candidates": zero_cand_s1_count,
        "avg": avg_cands,
        "median": median_cands,
        "p95": p95_cands,
        "max": max_cands_obs,
        "rule_counts": dict(rule_counts)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", action="store_true", help="Run candidate generation using tuned 7-pass index")
    parser.add_argument("--ranked", action="store_true", help="Apply relevance-based pre-ranking before capping")
    parser.add_argument("--num-norm", action="store_true", help="Save output to test_1k_candidates_ranked_num.parquet")
    args = parser.parse_args()

    s1_path = TRAIN_FILES["s1"]

    if args.num_norm:
        out_path = CANDIDATE_FILES["test_1k_candidates_ranked_num"]
        run_blocking_test_1k(s1_path, out_path, num_entities=1000, tuned=True, ranked=True)
    elif args.ranked:
        out_path = CANDIDATE_FILES["test_1k_candidates_ranked"]
        run_blocking_test_1k(s1_path, out_path, num_entities=1000, tuned=True, ranked=True)
    elif args.tuned:
        out_path = CANDIDATE_FILES["test_1k_candidates_tuned"]
        run_blocking_test_1k(s1_path, out_path, num_entities=1000, tuned=True, ranked=False)
    else:
        out_path = CANDIDATE_FILES["test_1k_candidates"]
        run_blocking_test_1k(s1_path, out_path, num_entities=1000, tuned=False, ranked=False)


if __name__ == "__main__":
    main()
