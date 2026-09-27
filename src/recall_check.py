import sys
import time
import argparse
import polars as pl
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "train"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"
for d in [CACHE_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"
CANDIDATES_PATH = CACHE_DIR / "train_pairs.parquet"
REPORT_PATH = OUTPUT_DIR / "recall_report.txt"

def load_ground_truth(gt_file: Path) -> pl.DataFrame:
    if not gt_file.exists():
        print(f"❌ Error: Ground truth file missing: {gt_file}")
        sys.exit(1)

    s1_list, cand_list = [], []
    with open(gt_file, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0]
            if len(parts) > 1 and parts[1].strip():
                for mid in parts[1].split(","):
                    mid = mid.strip()
                    if mid:
                        s1_list.append(s1_id)
                        cand_list.append(mid)

    return pl.DataFrame({
        "source1_entity_id": s1_list,
        "candidate_entity_id": cand_list
    })

def run_recall_check():
    print("=" * 80)
    print("STEP 3: CANDIDATE RECALL EVALUATION (GROUND TRUTH AUDIT)")
    print("=" * 80)
    t0 = time.time()

    if not CANDIDATES_PATH.exists():
        print(f"Candidates file {CANDIDATES_PATH} not found. Running blocking pass...")
        from src.block import block_split
        block_split("train")

    print(f"Loading candidate pairs from {CANDIDATES_PATH}...", flush=True)
    candidates = pl.read_parquet(CANDIDATES_PATH)
    total_candidates = len(candidates)

    print(f"Loading ground truth from {GT_PATH}...", flush=True)
    gt_df = load_ground_truth(GT_PATH)
    total_true_links = len(gt_df)

    s2_true = len(gt_df.filter(pl.col("candidate_entity_id").str.starts_with("S2-")))
    s3_true = len(gt_df.filter(pl.col("candidate_entity_id").str.starts_with("S3-")))

    s1_all = pl.read_parquet(CACHE_DIR / "train_source1.parquet", columns=["entity_id"]).rename({"entity_id": "source1_entity_id"})
    gt_counts = gt_df.group_by("source1_entity_id").agg(pl.len().alias("true_match_count"))
    s1_categories = s1_all.join(gt_counts, on="source1_entity_id", how="left").with_columns(
        pl.col("true_match_count").fill_null(0)
    )

    zero_match_s1 = s1_categories.filter(pl.col("true_match_count") == 0)
    single_match_s1 = s1_categories.filter(pl.col("true_match_count") == 1)
    multi_match_s1 = s1_categories.filter(pl.col("true_match_count") >= 2)

    matched = candidates.join(
        gt_df,
        on=["source1_entity_id", "candidate_entity_id"],
        how="inner"
    )

    total_found = len(matched)
    s2_found = len(matched.filter(pl.col("candidate_entity_id").str.starts_with("S2-")))
    s3_found = len(matched.filter(pl.col("candidate_entity_id").str.starts_with("S3-")))

    overall_recall = round(total_found / total_true_links * 100, 2)
    s2_recall = round(s2_found / s2_true * 100, 2) if s2_true > 0 else 0.0
    s3_recall = round(s3_found / s3_true * 100, 2) if s3_true > 0 else 0.0

    single_gt_links = gt_df.join(single_match_s1, on="source1_entity_id", how="inner")
    single_found = matched.join(single_match_s1, on="source1_entity_id", how="inner")
    single_recall = round(len(single_found) / len(single_gt_links) * 100, 2) if len(single_gt_links) > 0 else 0.0

    multi_gt_links = gt_df.join(multi_match_s1, on="source1_entity_id", how="inner")
    multi_found = matched.join(multi_match_s1, on="source1_entity_id", how="inner")
    multi_recall = round(len(multi_found) / len(multi_gt_links) * 100, 2) if len(multi_gt_links) > 0 else 0.0

    density = candidates.group_by("source1_entity_id").agg(pl.len().alias("count"))
    counts = density["count"].to_numpy()
    avg_cands = round(float(counts.mean()), 2)
    median_cands = float(density["count"].median())
    p95_cands = float(density["count"].quantile(0.95))
    p99_cands = float(density["count"].quantile(0.99))
    max_cands = int(counts.max())

    report = (
        "=" * 60 + "\n"
        "CANDIDATE RECALL EVALUATION SUMMARY\n"
        "=" * 60 + "\n"
        f"Total Candidate Pairs Generated: {total_candidates:,}\n"
        f"Average Candidates per S1    : {avg_cands}\n"
        f"Median Candidates per S1     : {median_cands}\n"
        f"P95 Candidates per S1        : {p95_cands}\n"
        f"P99 Candidates per S1        : {p99_cands}\n"
        f"Max Candidates per S1        : {max_cands}\n"
         + "-" * 60 + "\n"
        f"Total Ground Truth True Links: {total_true_links:,}\n"
        f"True Links Found in Candidates: {total_found:,}\n"
        f"OVERALL CANDIDATE RECALL     : {overall_recall}%\n"
        f"  - Source 2 Candidate Recall: {s2_recall}% ({s2_found:,} / {s2_true:,})\n"
        f"  - Source 3 Candidate Recall: {s3_recall}% ({s3_found:,} / {s3_true:,})\n"
         + "-" * 60 + "\n"
        "CANDIDATE RECALL BY GT MATCH CATEGORY:\n"
        f"  - Single-Match S1 Recall   : {single_recall}% ({len(single_found):,} / {len(single_gt_links):,})\n"
        f"  - Multi-Match S1 Recall    : {multi_recall}% ({len(multi_found):,} / {len(multi_gt_links):,})\n"
        f"Evaluation Runtime           : {round(time.time() - t0, 2)}s\n"
        + "=" * 60 + "\n"
    )

    print("\n" + report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {REPORT_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Recall check for candidate pairs.")
    parser.add_argument("split", nargs="?", default="train", choices=["train"], help="Split to evaluate")
    args = parser.parse_args()
    run_recall_check()

if __name__ == "__main__":
    main()