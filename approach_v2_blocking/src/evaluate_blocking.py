"""
Approach V2 Candidate Recall Evaluator (Baseline vs Tuned vs Tuned+Ranked)
Evaluates candidate recall against ground truth labels.
Saves outputs to candidate_recall_1k_ranked.json and candidate_recall_1k_ranked.txt.
Explicitly verifies known truncation failure cases.
"""

import csv
import json
import sys
import time
import argparse
import polars as pl
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

csv.field_size_limit(2147483647)

# Add approach root to sys.path
SRC_DIR = Path(__file__).resolve().parent
APPROACH_ROOT = SRC_DIR.parent
if str(APPROACH_ROOT) not in sys.path:
    sys.path.insert(0, str(APPROACH_ROOT))

from src.config import TRAIN_FILES, CANDIDATE_FILES, EXPERIMENTS_DIR


def evaluate_blocking_recall(tuned: bool = False, ranked: bool = False, num_norm: bool = False):
    if tuned and ranked and num_norm:
        label_mode = "TUNED + RELEVANCE RANKING (NUMERIC NORMALIZED)"
        cand_parquet = CANDIDATE_FILES["test_1k_candidates_ranked_num"]
        prefix_fn = "candidate_recall_1k_ranked_numeric_normalized"
    elif tuned and ranked:
        label_mode = "TUNED + RELEVANCE RANKING"
        cand_parquet = CANDIDATE_FILES["test_1k_candidates_ranked"]
        prefix_fn = "candidate_recall_1k_ranked"
    elif tuned:
        label_mode = "TUNED (UNRANKED)"
        cand_parquet = CANDIDATE_FILES["test_1k_candidates_tuned"]
        prefix_fn = "candidate_recall_1k_tuned"
    else:
        label_mode = "BASELINE (4 PASSES)"
        cand_parquet = CANDIDATE_FILES["test_1k_candidates"]
        prefix_fn = "candidate_recall_1k"

    print("=" * 80)
    print(f"APPROACH V2: CANDIDATE RECALL EVALUATION ({label_mode})")
    print("=" * 80)
    t0 = time.time()

    gt_tsv = TRAIN_FILES["gt"]
    s1_tsv = TRAIN_FILES["s1"]

    if not cand_parquet.exists():
        print(f"❌ Error: Candidate file not found at {cand_parquet}")
        return

    # 1. Load the exact 1,000 tested S1 entity IDs in order
    print("Loading tested S1 entity IDs...", flush=True)
    s1_tested_ids = []
    with open(s1_tsv, mode="r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_tested_ids.append(row["entity_id"].strip())
            if len(s1_tested_ids) >= 1000:
                break

    set_s1_tested = set(s1_tested_ids)

    # 2. Load Ground Truth for these 1,000 S1 entities
    print("Loading Ground Truth labels...", flush=True)
    gt_map = {s1_id: set() for s1_id in s1_tested_ids}
    total_gt_links = 0

    with open(gt_tsv, mode="r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row.get("source1_entity_id", "").strip()
            if s1_id in set_s1_tested:
                raw_matches = row.get("matched_entity_ids", "").strip()
                if raw_matches:
                    matches = {m.strip() for m in raw_matches.split(",") if m.strip()}
                    gt_map[s1_id] = matches
                    total_gt_links += len(matches)

    # 3. Load Candidates from Parquet
    print(f"Loading candidates from {cand_parquet.name}...", flush=True)
    df_cands = pl.read_parquet(cand_parquet)

    all_rule_names = ["exact_name", "name_token", "address_token", "name_prefix"]
    if tuned or ranked:
        all_rule_names.extend(["char_ngram", "postal_code", "relaxed_token"])

    union_cand_map = {s1_id: {} for s1_id in s1_tested_ids}
    pass_cand_map = {rule: {s1_id: set() for s1_id in s1_tested_ids} for rule in all_rule_names}

    for row in df_cands.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        passes_str = row["blocking_passes"]
        passes_list = passes_str.split(",")

        if s1_id in set_s1_tested:
            union_cand_map[s1_id][cand_id] = passes_str
            for p in passes_list:
                if p in pass_cand_map:
                    pass_cand_map[p][s1_id].add(cand_id)

    # 4. Compute Overall & Entity-Level Metrics
    zero_gt_s1_count = 0
    all_retrieved_s1_count = 0
    partial_retrieved_s1_count = 0
    zero_retrieved_s1_count = 0

    total_gt_retrieved_union = 0

    all_retrieved_examples = []
    partial_retrieved_examples = []
    zero_retrieved_examples = []

    for s1_id in s1_tested_ids:
        true_set = gt_map[s1_id]
        n_true = len(true_set)

        if n_true == 0:
            zero_gt_s1_count += 1
            continue

        cand_dict = union_cand_map[s1_id]
        cand_set = set(cand_dict.keys())
        retrieved_set = true_set & cand_set
        missed_set = true_set - cand_set
        n_retrieved = len(retrieved_set)
        total_gt_retrieved_union += n_retrieved

        example_data = {
            "s1_entity_id": s1_id,
            "ground_truth_count": n_true,
            "retrieved_count": n_retrieved,
            "ground_truth_matches": sorted(list(true_set)),
            "retrieved_true_matches": sorted(list(retrieved_set)),
            "missed_true_matches": sorted(list(missed_set)),
            "generated_candidates": [
                {"candidate_entity_id": cid, "blocking_passes": passes}
                for cid, passes in list(cand_dict.items())[:10]
            ]
        }

        if n_retrieved == n_true:
            all_retrieved_s1_count += 1
            if len(all_retrieved_examples) < 3:
                all_retrieved_examples.append(example_data)
        elif n_retrieved > 0:
            partial_retrieved_s1_count += 1
            if len(partial_retrieved_examples) < 3:
                partial_retrieved_examples.append(example_data)
        else:
            zero_retrieved_s1_count += 1
            if len(zero_retrieved_examples) < 3:
                zero_retrieved_examples.append(example_data)

    active_s1_count = len(s1_tested_ids) - zero_gt_s1_count
    overall_recall_pct = round(total_gt_retrieved_union / total_gt_links * 100, 2) if total_gt_links > 0 else 0.0
    missed_gt_links = total_gt_links - total_gt_retrieved_union

    # 5. Verify Specific Known Truncation Examples
    target_verification = []

    known_targets = [
        ("S1-748243940", "S3-749582950"),
        ("S1-666435696", "S3-211915957")
    ]

    for s1_target, cand_target in known_targets:
        is_retrieved = cand_target in union_cand_map.get(s1_target, {})
        passes_trig = union_cand_map.get(s1_target, {}).get(cand_target, "N/A")
        target_verification.append({
            "s1_entity_id": s1_target,
            "target_candidate_id": cand_target,
            "retrieved_in_top100": is_retrieved,
            "blocking_passes": passes_trig
        })

    # 6. Compute Per-Pass Metrics
    pass_recall_stats = {}
    for rule_name, rule_cand_map in pass_cand_map.items():
        rule_retrieved_links = 0
        rule_active_s1_entities = 0

        for s1_id in s1_tested_ids:
            true_set = gt_map[s1_id]
            if true_set:
                r_set = true_set & rule_cand_map[s1_id]
                rule_retrieved_links += len(r_set)
                if len(r_set) > 0:
                    rule_active_s1_entities += 1

        rule_recall_pct = round(rule_retrieved_links / total_gt_links * 100, 2) if total_gt_links > 0 else 0.0
        pass_recall_stats[rule_name] = {
            "retrieved_gt_links": rule_retrieved_links,
            "recall_pct": rule_recall_pct,
            "s1_entities_with_at_least_one_true_match": rule_active_s1_entities,
            "s1_entities_pct": round(rule_active_s1_entities / active_s1_count * 100, 2) if active_s1_count > 0 else 0.0
        }

    t_elapsed = round(time.time() - t0, 2)

    # 7. Construct JSON Payload
    json_results = {
        "summary": {
            "mode": label_mode,
            "total_s1_entities_evaluated": len(s1_tested_ids),
            "s1_entities_with_zero_gt_matches": zero_gt_s1_count,
            "s1_entities_with_active_gt_matches": active_s1_count,
            "total_gt_positive_matches": total_gt_links,
            "total_gt_matches_retrieved": total_gt_retrieved_union,
            "total_gt_matches_missed": missed_gt_links,
            "overall_candidate_recall_pct": overall_recall_pct,
            "entity_recall_breakdown": {
                "all_gt_matches_retrieved_count": all_retrieved_s1_count,
                "all_gt_matches_retrieved_pct": round(all_retrieved_s1_count / active_s1_count * 100, 2),
                "partial_gt_matches_retrieved_count": partial_retrieved_s1_count,
                "partial_gt_matches_retrieved_pct": round(partial_retrieved_s1_count / active_s1_count * 100, 2),
                "zero_gt_matches_retrieved_count": zero_retrieved_s1_count,
                "zero_gt_matches_retrieved_pct": round(zero_retrieved_s1_count / active_s1_count * 100, 2)
            }
        },
        "target_verifications": target_verification,
        "per_blocking_pass": pass_recall_stats,
        "union_all_passes": {
            "total_gt_retrieved": total_gt_retrieved_union,
            "overall_recall_pct": overall_recall_pct,
            "s1_entities_with_at_least_one_true_match": all_retrieved_s1_count + partial_retrieved_s1_count,
            "s1_entities_pct": round((all_retrieved_s1_count + partial_retrieved_s1_count) / active_s1_count * 100, 2)
        }
    }

    out_json = EXPERIMENTS_DIR / f"{prefix_fn}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)

    # 8. Construct Human-Readable Text Report
    out_txt = EXPERIMENTS_DIR / f"{prefix_fn}.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"APPROACH V2: CANDIDATE RECALL EVALUATION REPORT ({label_mode})\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. OVERALL METRICS:\n")
        f.write(f"  - Total S1 Entities Evaluated          : {len(s1_tested_ids):,}\n")
        f.write(f"  - S1 Entities with 0 GT Matches        : {zero_gt_s1_count:,} ({round(zero_gt_s1_count/len(s1_tested_ids)*100, 2)}%)\n")
        f.write(f"  - S1 Entities with >=1 GT Matches      : {active_s1_count:,}\n")
        f.write(f"  - Total Ground-Truth Positive Matches  : {total_gt_links:,}\n")
        f.write(f"  - Total GT Matches Retrieved (Union)   : {total_gt_retrieved_union:,}\n")
        f.write(f"  - Total GT Matches Missed              : {missed_gt_links:,}\n")
        f.write(f"  - OVERALL CANDIDATE RECALL %           : {overall_recall_pct}%\n\n")

        f.write("2. ENTITY-LEVEL RECALL BREAKDOWN (for S1 with >=1 GT link):\n")
        f.write(f"  - ALL GT Matches Retrieved (Complete) : {all_retrieved_s1_count:,} entities ({round(all_retrieved_s1_count/active_s1_count*100, 2)}%)\n")
        f.write(f"  - PARTIAL GT Matches Retrieved        : {partial_retrieved_s1_count:,} entities ({round(partial_retrieved_s1_count/active_s1_count*100, 2)}%)\n")
        f.write(f"  - ZERO GT Matches Retrieved           : {zero_retrieved_s1_count:,} entities ({round(zero_retrieved_s1_count/active_s1_count*100, 2)}%)\n\n")

        f.write("3. KNOWN TRUNCATION CASE VERIFICATIONS:\n")
        for v in target_verification:
            status_str = "SUCCESSFULLY RETRIEVED IN TOP-100" if v["retrieved_in_top100"] else "FAILED / NOT IN TOP-100"
            f.write(f"  - S1 [{v['s1_entity_id']}] -> GT [{v['target_candidate_id']}] : {status_str} (Passes: {v['blocking_passes']})\n")
        f.write("\n")

        f.write("4. PER-BLOCKING-PASS METRICS:\n")
        f.write(f"  - Union of All Passes                 : {total_gt_retrieved_union:,} / {total_gt_links:,} links ({overall_recall_pct}%) | {all_retrieved_s1_count+partial_retrieved_s1_count:,} S1 entities\n")
        for rule_name, stat in pass_recall_stats.items():
            f.write(f"  - Rule [{rule_name:15s}]          : {stat['retrieved_gt_links']:,} / {total_gt_links:,} links ({stat['recall_pct']}%) | {stat['s1_entities_with_at_least_one_true_match']:,} S1 entities ({stat['s1_entities_pct']}%)\n")
        f.write("\n")

        f.write("=" * 80 + "\n")

    print("\n" + "=" * 80)
    print(f"CANDIDATE RECALL EVALUATION SUMMARY ({label_mode})")
    print("=" * 80)
    print(f"Total S1 Entities Evaluated          : {len(s1_tested_ids):,}")
    print(f"S1 Entities with 0 GT Links          : {zero_gt_s1_count:,} ({round(zero_gt_s1_count/len(s1_tested_ids)*100, 2)}%)")
    print(f"S1 Entities with >=1 GT Link         : {active_s1_count:,}")
    print("-" * 60)
    print(f"Total Ground-Truth Positive Matches  : {total_gt_links:,}")
    print(f"Ground-Truth Matches Retrieved       : {total_gt_retrieved_union:,}")
    print(f"Ground-Truth Matches Missed          : {missed_gt_links:,}")
    print(f"OVERALL CANDIDATE RECALL % (UNION)   : {overall_recall_pct}%")
    print("-" * 60)
    print("ENTITY-LEVEL RECALL BREAKDOWN:")
    print(f"  - ALL GT Matches Retrieved (Complete): {all_retrieved_s1_count:,} entities ({round(all_retrieved_s1_count/active_s1_count*100, 2)}%)")
    print(f"  - PARTIAL GT Matches Retrieved       : {partial_retrieved_s1_count:,} entities ({round(partial_retrieved_s1_count/active_s1_count*100, 2)}%)")
    print(f"  - ZERO GT Matches Retrieved          : {zero_retrieved_s1_count:,} entities ({round(zero_retrieved_s1_count/active_s1_count*100, 2)}%)")
    print("-" * 60)
    print("KNOWN TRUNCATION CASE VERIFICATIONS:")
    for v in target_verification:
        status_str = "SUCCESS (Retrieved in Top-100)" if v["retrieved_in_top100"] else "FAILED"
        print(f"  - {v['s1_entity_id']} -> {v['target_candidate_id']} : {status_str} (Passes: {v['blocking_passes']})")
    print("-" * 60)
    print(f"Saved evaluation JSON report to       : {out_json.name}")
    print(f"Saved evaluation TXT report to        : {out_txt.name}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", action="store_true", help="Evaluate tuned 7-pass candidates")
    parser.add_argument("--ranked", action="store_true", help="Evaluate relevance-ranked candidates")
    parser.add_argument("--num-norm", action="store_true", help="Evaluate numeric-normalized candidates")
    args = parser.parse_args()

    evaluate_blocking_recall(tuned=args.tuned, ranked=args.ranked, num_norm=args.num_norm)


if __name__ == "__main__":
    main()
