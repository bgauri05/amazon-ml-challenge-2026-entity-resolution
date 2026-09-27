"""
Approach V2 Data Inspection Script
Inspects raw training TSVs using streaming csv.DictReader.
Calculates row counts, missing values, uniqueness, duplicate stats, text lengths, country distributions,
and ground truth match distributions.
"""

import csv
import sys
import time
from collections import Counter
from pathlib import Path

# Set large field limit for csv parsing
csv.field_size_limit(2147483647)

# Add approach root to sys.path
SRC_DIR = Path(__file__).resolve().parent
APPROACH_ROOT = SRC_DIR.parent
if str(APPROACH_ROOT) not in sys.path:
    sys.path.insert(0, str(APPROACH_ROOT))

from src.config import TRAIN_FILES


def inspect_source_tsv(file_path: Path, label: str) -> dict:
    print("\n" + "=" * 80)
    print(f"INSPECTING: {label} ({file_path.name})")
    print("=" * 80)
    t0 = time.time()

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return {}

    row_count = 0
    columns = []

    missing_name = 0
    missing_address = 0
    missing_country = 0

    total_name_len = 0
    total_address_len = 0

    name_counts = Counter()
    address_counts = Counter()
    country_counts = Counter()

    with open(file_path, mode="r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        columns = reader.fieldnames or []

        for row in reader:
            row_count += 1

            name = (row.get("business_name") or "").strip()
            address = (row.get("business_address") or "").strip()
            country = (row.get("country") or "").strip()

            # Missing value checks
            if not name:
                missing_name += 1
            else:
                total_name_len += len(name)
                name_counts[name] += 1

            if not address:
                missing_address += 1
            else:
                total_address_len += len(address)
                address_counts[address] += 1

            if not country:
                missing_country += 1
            else:
                country_counts[country] += 1

            if row_count % 200_000 == 0:
                print(f"  ...processed {row_count:,} rows", flush=True)

    t_elapsed = round(time.time() - t0, 2)

    # Name metrics
    distinct_names = len(name_counts)
    duplicate_name_types = sum(1 for cnt in name_counts.values() if cnt > 1)
    duplicate_name_rows = sum(cnt for cnt in name_counts.values() if cnt > 1)
    avg_name_len = round(total_name_len / (row_count - missing_name), 2) if (row_count - missing_name) > 0 else 0.0

    # Address metrics
    distinct_addresses = len(address_counts)
    duplicate_address_types = sum(1 for cnt in address_counts.values() if cnt > 1)
    duplicate_address_rows = sum(cnt for cnt in address_counts.values() if cnt > 1)
    avg_address_len = round(total_address_len / (row_count - missing_address), 2) if (row_count - missing_address) > 0 else 0.0

    # Top 15 countries
    top15_countries = country_counts.most_common(15)

    print(f"Finished in {t_elapsed}s.")
    print(f"Total Rows              : {row_count:,}")
    print(f"Columns                 : {columns}")
    print("-" * 60)
    print("MISSING VALUES:")
    print(f"  - Missing Name        : {missing_name:,} ({round(missing_name/row_count*100, 4)}%)")
    print(f"  - Missing Address     : {missing_address:,} ({round(missing_address/row_count*100, 4)}%)")
    print(f"  - Missing Country     : {missing_country:,} ({round(missing_country/row_count*100, 4)}%)")
    print("-" * 60)
    print("BUSINESS NAME METRICS:")
    print(f"  - Unique Names        : {distinct_names:,}")
    print(f"  - Duplicate Name Types: {duplicate_name_types:,} (appearing in {duplicate_name_rows:,} total rows)")
    print(f"  - Avg Name Length     : {avg_name_len} chars")
    print("-" * 60)
    print("BUSINESS ADDRESS METRICS:")
    print(f"  - Unique Addresses    : {distinct_addresses:,}")
    print(f"  - Duplicate Address Types: {duplicate_address_types:,} (appearing in {duplicate_address_rows:,} total rows)")
    print(f"  - Avg Address Length  : {avg_address_len} chars")
    print("-" * 60)
    print("TOP 15 COUNTRIES:")
    for ctry, cnt in top15_countries:
        print(f"  - {ctry or '<EMPTY>'}: {cnt:,} ({round(cnt/row_count*100, 2)}%)")

    return {
        "row_count": row_count,
        "columns": columns,
        "missing_name": missing_name,
        "missing_address": missing_address,
        "missing_country": missing_country,
        "distinct_names": distinct_names,
        "duplicate_name_types": duplicate_name_types,
        "duplicate_name_rows": duplicate_name_rows,
        "distinct_addresses": distinct_addresses,
        "duplicate_address_types": duplicate_address_types,
        "duplicate_address_rows": duplicate_address_rows,
        "avg_name_len": avg_name_len,
        "avg_address_len": avg_address_len,
        "top15_countries": top15_countries,
        "s1_ids": set(name_counts.keys()) if "s1" in label.lower() else set()
    }


def inspect_ground_truth(file_path: Path, s1_all_ids: set) -> dict:
    print("\n" + "=" * 80)
    print(f"INSPECTING: GROUND TRUTH ({file_path.name})")
    print("=" * 80)
    t0 = time.time()

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return {}

    gt_s1_match_counts = {}
    gt_rows = 0

    with open(file_path, mode="r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gt_rows += 1
            s1_id = row.get("source1_entity_id", "").strip()
            raw_matches = row.get("matched_entity_ids", "").strip()

            if raw_matches:
                matches = [m.strip() for m in raw_matches.split(",") if m.strip()]
            else:
                matches = []

            gt_s1_match_counts[s1_id] = len(matches)

    # Combine with all S1 IDs if available
    all_s1_counts = Counter()

    if s1_all_ids:
        total_s1 = len(s1_all_ids)
        for s1_id in s1_all_ids:
            cnt = gt_s1_match_counts.get(s1_id, 0)
            all_s1_counts[cnt] += 1
    else:
        total_s1 = len(gt_s1_match_counts)
        for cnt in gt_s1_match_counts.values():
            all_s1_counts[cnt] += 1

    t_elapsed = round(time.time() - t0, 2)

    print(f"Finished in {t_elapsed}s.")
    print(f"Total Ground Truth Rows Listed: {gt_rows:,}")
    print(f"Total Unique Source-1 Entities : {total_s1:,}")
    print("-" * 60)
    print("MATCH COUNT DISTRIBUTION PER SOURCE-1 ENTITY:")
    sorted_dist = sorted(all_s1_counts.items(), key=lambda x: x[0])
    for match_cnt, s1_cnt in sorted_dist:
        print(f"  - {match_cnt} matched entities : {s1_cnt:,} S1 entities ({round(s1_cnt / total_s1 * 100, 2)}%)")

    return {
        "gt_rows": gt_rows,
        "total_s1": total_s1,
        "distribution": dict(sorted_dist)
    }


def collect_s1_ids(file_path: Path) -> set:
    s1_ids = set()
    with open(file_path, mode="r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row.get("entity_id", "").strip()
            if s1_id:
                s1_ids.add(s1_id)
    return s1_ids


def main():
    print("=" * 80)
    print("STARTING APPROACH V2 STREAMING DATA INSPECTION")
    print("=" * 80)

    # 1. Source 1
    s1_stats = inspect_source_tsv(TRAIN_FILES["s1"], "Source 1")

    # 2. Source 2
    s2_stats = inspect_source_tsv(TRAIN_FILES["s2"], "Source 2")

    # 3. Source 3
    s3_stats = inspect_source_tsv(TRAIN_FILES["s3"], "Source 3")

    # Collect S1 IDs for exact match distribution calculation
    print("\nCollecting S1 entity IDs for Ground Truth mapping...", flush=True)
    s1_all_ids = collect_s1_ids(TRAIN_FILES["s1"])

    # 4. Ground Truth
    gt_stats = inspect_ground_truth(TRAIN_FILES["gt"], s1_all_ids)

    print("\n" + "=" * 80)
    print("DATA INSPECTION COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
