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
TEST_DIR = DATA_DIR / "test"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"

for d in [CACHE_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LEGAL_REGEX = (
    r"\b("
    r"private limited|pvt ltd|pvt\. ltd\.|pvt|ltd|limited|"
    r"incorporated|inc\.|inc|corporation|corp\.|corp|"
    r"limited liability company|llc\.|llc|llp\.|llp|gmbh|sarl|sas|sa|plc"
    r")\b"
)

def normalize_dataset_tsv(tsv_path: Path, parquet_path: Path, dataset_label: str):
    if not tsv_path.exists():
        print(f"❌ Error: Source file not found: {tsv_path}")
        return None

    print(f"[{dataset_label}] Normalizing {tsv_path.name} -> {parquet_path.name}...", flush=True)
    t0 = time.time()

    df = pl.read_csv(
        tsv_path,
        separator="\t",
        has_header=True,
        quote_char=None,
        ignore_errors=True
    )

    total_rows = len(df)

    df_cache = df.with_columns([
        pl.col("business_name")
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("name_norm"),

        pl.col("business_address")
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("address_norm"),

        pl.col("country")
        .fill_null("")
        .str.strip_chars()
        .alias("country_norm"),

        pl.col("business_address")
        .fill_null("")
        .str.extract(r"\b(\d{5,6})\b", 1)
        .fill_null("")
        .alias("postal_code"),

        pl.col("business_address")
        .fill_null("")
        .str.extract_all(r"\d+")
        .alias("numbers")
    ]).with_columns([
        pl.col("name_norm")
        .str.replace_all(LEGAL_REGEX, "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("name_core"),

        pl.col("name_norm")
        .str.split(" ")
        .list.eval(pl.element().filter(pl.element().str.len_bytes() >= 2))
        .alias("name_tokens"),

        pl.col("address_norm")
        .str.split(" ")
        .list.eval(pl.element().filter(pl.element().str.len_bytes() >= 2))
        .alias("address_tokens")
    ])

    df_cache.write_parquet(parquet_path, compression="snappy")

    t_delta = round(time.time() - t0, 2)
    in_mb = round(tsv_path.stat().st_size / (1024 * 1024), 2)
    out_mb = round(parquet_path.stat().st_size / (1024 * 1024), 2)

    print(f"  [DONE] {dataset_label}: {total_rows:,} rows processed in {t_delta}s | Input TSV: {in_mb} MB | Parquet Cache: {out_mb} MB", flush=True)
    return total_rows

def normalize_split(split: str):
    print("=" * 80)
    print(f"STEP 1: TEXT NORMALIZATION ({split.upper()})")
    print("=" * 80)
    t_start = time.time()

    if split in ("train", "all"):
        normalize_dataset_tsv(TRAIN_DIR / "train_source1.tsv", CACHE_DIR / "train_source1.parquet", "train_source1")
        normalize_dataset_tsv(TRAIN_DIR / "train_source2.tsv", CACHE_DIR / "train_source2.parquet", "train_source2")
        normalize_dataset_tsv(TRAIN_DIR / "train_source3.tsv", CACHE_DIR / "train_source3.parquet", "train_source3")

    if split in ("test", "all"):
        normalize_dataset_tsv(TEST_DIR / "test_source1.tsv", CACHE_DIR / "test_source1.parquet", "test_source1")
        normalize_dataset_tsv(TEST_DIR / "test_source2.tsv", CACHE_DIR / "test_source2.parquet", "test_source2")
        normalize_dataset_tsv(TEST_DIR / "test_source3.tsv", CACHE_DIR / "test_source3.parquet", "test_source3")

    print(f"\n[NORMALIZATION COMPLETED IN {round(time.time() - t_start, 2)}s]")

def main():
    parser = argparse.ArgumentParser(description="Normalize Amazon ML dataset TSVs into Parquet cache.")
    parser.add_argument(
        "split",
        nargs="?",
        default="all",
        choices=["train", "test", "all"],
        help="Dataset split to normalize (train, test, or all)"
    )
    args = parser.parse_args()
    normalize_split(args.split)

if __name__ == "__main__":
    main()