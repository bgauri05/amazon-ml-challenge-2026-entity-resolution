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

CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"
for d in [CACHE_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GENERIC_STOP_WORDS = {
    "company", "services", "solutions", "global", "international",
    "traders", "enterprises", "group", "holdings", "industries",
    "market", "store", "shop", "center", "centre", "tech", "technologies",
    "pvt", "ltd", "inc", "corp", "llc", "gmbh", "limited", "private"
}

MAX_NAME_BLOCK_FREQ = 50   # Max frequency for exact name/core blocks
MAX_TOKEN_BLOCK_FREQ = 30  # Max frequency for first-token blocks
MAX_POSTAL_BLOCK_FREQ = 30 # Max frequency for postal code blocks

def generate_candidates_for_split(s1_parquet: Path, s2_parquet: Path, s3_parquet: Path, split_name: str) -> pl.DataFrame:
    print(f"\n[{split_name}] Generating candidate pairs...", flush=True)
    t0 = time.time()

    if not (s1_parquet.exists() and s2_parquet.exists() and s3_parquet.exists()):
        print(f"❌ Error: Required parquet cache files missing for {split_name}. Run python src/normalize.py {split_name.lower()} first.")
        sys.exit(1)

    cols = ["entity_id", "country_norm", "name_norm", "name_core", "postal_code"]
    s1 = pl.read_parquet(s1_parquet, columns=cols)
    s2 = pl.read_parquet(s2_parquet, columns=cols)
    s3 = pl.read_parquet(s3_parquet, columns=cols)

    def extract_first_token(df_in: pl.DataFrame) -> pl.DataFrame:
        return df_in.with_columns(
            pl.col("name_core")
            .str.split(" ")
            .list.eval(
                pl.element().filter(
                    (pl.element().str.len_bytes() >= 4) & (~pl.element().is_in(list(GENERIC_STOP_WORDS)))
                )
            )
            .list.first()
            .fill_null("")
            .alias("first_token")
        )

    s1 = extract_first_token(s1)
    s2 = extract_first_token(s2)
    s3 = extract_first_token(s3)

    candidates_list = []

    for s_target, source_label in [(s2, "S2"), (s3, "S3")]:
        # Pass 1: Core Name
        s1_c1 = s1.filter(pl.col("name_core").str.len_bytes() >= 3).group_by(["country_norm", "name_core"]).agg(pl.len().alias("f1"))
        st_c1 = s_target.filter(pl.col("name_core").str.len_bytes() >= 3).group_by(["country_norm", "name_core"]).agg(pl.len().alias("f2"))
        valid1 = s1_c1.join(st_c1, on=["country_norm", "name_core"]).filter(
            (pl.col("f1") <= MAX_NAME_BLOCK_FREQ) & (pl.col("f2") <= MAX_NAME_BLOCK_FREQ)
        ).select(["country_norm", "name_core"])

        pass1 = s1.join(valid1, on=["country_norm", "name_core"]).join(
            s_target.join(valid1, on=["country_norm", "name_core"]),
            on=["country_norm", "name_core"],
            suffix="_cand"
        ).select([
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("entity_id_cand").alias("candidate_entity_id"),
            pl.lit(source_label).alias("candidate_source"),
            pl.lit("core_name").alias("blocking_method")
        ])
        candidates_list.append(pass1)

        # Pass 2: Normalized Name
        s1_c2 = s1.filter(pl.col("name_norm").str.len_bytes() >= 3).group_by(["country_norm", "name_norm"]).agg(pl.len().alias("f1"))
        st_c2 = s_target.filter(pl.col("name_norm").str.len_bytes() >= 3).group_by(["country_norm", "name_norm"]).agg(pl.len().alias("f2"))
        valid2 = s1_c2.join(st_c2, on=["country_norm", "name_norm"]).filter(
            (pl.col("f1") <= MAX_NAME_BLOCK_FREQ) & (pl.col("f2") <= MAX_NAME_BLOCK_FREQ)
        ).select(["country_norm", "name_norm"])

        pass2 = s1.join(valid2, on=["country_norm", "name_norm"]).join(
            s_target.join(valid2, on=["country_norm", "name_norm"]),
            on=["country_norm", "name_norm"],
            suffix="_cand"
        ).select([
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("entity_id_cand").alias("candidate_entity_id"),
            pl.lit(source_label).alias("candidate_source"),
            pl.lit("norm_name").alias("blocking_method")
        ])
        candidates_list.append(pass2)

        # Pass 3: Distinctive First Token
        s1_c3 = s1.filter(pl.col("first_token").str.len_bytes() >= 4).group_by(["country_norm", "first_token"]).agg(pl.len().alias("f1"))
        st_c3 = s_target.filter(pl.col("first_token").str.len_bytes() >= 4).group_by(["country_norm", "first_token"]).agg(pl.len().alias("f2"))
        valid3 = s1_c3.join(st_c3, on=["country_norm", "first_token"]).filter(
            (pl.col("f1") <= MAX_TOKEN_BLOCK_FREQ) & (pl.col("f2") <= MAX_TOKEN_BLOCK_FREQ)
        ).select(["country_norm", "first_token"])

        pass3 = s1.join(valid3, on=["country_norm", "first_token"]).join(
            s_target.join(valid3, on=["country_norm", "first_token"]),
            on=["country_norm", "first_token"],
            suffix="_cand"
        ).filter(pl.col("entity_id") != pl.col("entity_id_cand")).select([
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("entity_id_cand").alias("candidate_entity_id"),
            pl.lit(source_label).alias("candidate_source"),
            pl.lit("first_token").alias("blocking_method")
        ])
        candidates_list.append(pass3)

        # Pass 4: Postal Code
        s1_c4 = s1.filter(pl.col("postal_code").str.len_bytes() >= 4).group_by(["country_norm", "postal_code"]).agg(pl.len().alias("f1"))
        st_c4 = s_target.filter(pl.col("postal_code").str.len_bytes() >= 4).group_by(["country_norm", "postal_code"]).agg(pl.len().alias("f2"))
        valid4 = s1_c4.join(st_c4, on=["country_norm", "postal_code"]).filter(
            (pl.col("f1") <= MAX_POSTAL_BLOCK_FREQ) & (pl.col("f2") <= MAX_POSTAL_BLOCK_FREQ)
        ).select(["country_norm", "postal_code"])

        pass4 = s1.join(valid4, on=["country_norm", "postal_code"]).join(
            s_target.join(valid4, on=["country_norm", "postal_code"]),
            on=["country_norm", "postal_code"],
            suffix="_cand"
        ).filter(pl.col("entity_id") != pl.col("entity_id_cand")).select([
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("entity_id_cand").alias("candidate_entity_id"),
            pl.lit(source_label).alias("candidate_source"),
            pl.lit("postal_code").alias("blocking_method")
        ])
        candidates_list.append(pass4)

    all_cands = pl.concat(candidates_list)
    unique_cands = all_cands.group_by(
        ["source1_entity_id", "candidate_entity_id", "candidate_source"]
    ).agg(
        pl.col("blocking_method").unique().sort().str.join(",").alias("blocking_methods")
    )

    t_delta = round(time.time() - t0, 2)
    total_cnt = len(unique_cands)
    s2_cnt = len(unique_cands.filter(pl.col("candidate_source") == "S2"))
    s3_cnt = len(unique_cands.filter(pl.col("candidate_source") == "S3"))

    print(f"  [DONE] {split_name}: {total_cnt:,} candidate pairs generated in {t_delta}s (S2: {s2_cnt:,}, S3: {s3_cnt:,})", flush=True)

    return unique_cands

def block_split(split: str):
    print("=" * 80)
    print(f"STEP 2: MULTI-PASS CANDIDATE BLOCKING ({split.upper()})")
    print("=" * 80)
    t_start = time.time()

    if split in ("train", "all"):
        c_train = generate_candidates_for_split(
            CACHE_DIR / "train_source1.parquet",
            CACHE_DIR / "train_source2.parquet",
            CACHE_DIR / "train_source3.parquet",
            "Train Set"
        )
        out_pairs = CACHE_DIR / "train_pairs.parquet"
        c_train.write_parquet(out_pairs, compression="snappy")
        print(f"Saved candidate pairs to {out_pairs}")

    if split in ("test", "all"):
        c_test = generate_candidates_for_split(
            CACHE_DIR / "test_source1.parquet",
            CACHE_DIR / "test_source2.parquet",
            CACHE_DIR / "test_source3.parquet",
            "Test Set"
        )
        out_pairs = CACHE_DIR / "test_pairs.parquet"
        c_test.write_parquet(out_pairs, compression="snappy")
        print(f"Saved candidate pairs to {out_pairs}")

    print(f"\n[BLOCKING COMPLETED IN {round(time.time() - t_start, 2)}s]")

def main():
    parser = argparse.ArgumentParser(description="Multi-pass candidate generation & blocking.")
    parser.add_argument(
        "split",
        nargs="?",
        default="train",
        choices=["train", "test", "all"],
        help="Dataset split to run blocking on (train, test, or all)"
    )
    args = parser.parse_args()
    block_split(args.split)

if __name__ == "__main__":
    main()