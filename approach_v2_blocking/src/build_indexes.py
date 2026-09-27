"""
Ultra-Fast Inverted Index Builder for Approach V2 (Baseline & Tuned)
Streams S2 and S3 TSVs, extracts baseline (1-4) or tuned (1-7) blocking keys,
inserts into SQLite disk-backed index with frequency pruning (MAX_BLOCK_SIZE).
Updated with numeric address token leading zero normalization.
"""

import csv
import re
import sys
import time
import sqlite3
import unicodedata
from collections import Counter
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
    MAX_BLOCK_SIZE,
    NAME_STOP_WORDS,
    ADDRESS_STOP_WORDS,
    COMMON_CHAR_NGRAMS,
    RELAXED_STOP_WORDS
)

RE_NON_WORD = re.compile(r"[^\w\s]")
RE_SPACES = re.compile(r"\s+")
RE_POSTAL = re.compile(r"\b\d{5,6}\b")

LEGAL_SUFFIX_REGEX = re.compile(
    r"\b("
    r"private limited|pvt ltd|pvt\. ltd\.|pvt|private|"
    r"limited|ltd\.|ltd|"
    r"incorporated|inc\.|inc|"
    r"corporation|corp\.|corp|"
    r"limited liability company|llc\.|llc|llp\.|llp|"
    r"company|co\.|co|gmbh|sarl|sas|sa|plc|"
    r"प्राइवेट लिमिटेड|प्राइवेट|लिमिटेड"
    r")\b",
    flags=re.IGNORECASE
)


def fast_normalize_name(name: str) -> str:
    if not name:
        return ""
    text = unicodedata.normalize("NFKC", str(name)).lower().replace("&", " and ")
    text = LEGAL_SUFFIX_REGEX.sub(" ", text)
    text = RE_NON_WORD.sub(" ", text)
    text = LEGAL_SUFFIX_REGEX.sub(" ", text)
    return RE_SPACES.sub(" ", text).strip()


def fast_normalize_address(address: str) -> str:
    if not address or not isinstance(address, str):
        return ""
    text = unicodedata.normalize("NFKC", address).lower()
    text = RE_NON_WORD.sub(" ", text)
    tokens = text.split()
    norm_tokens = [str(int(t)) if (t.isdigit() and len(t) > 1) else t for t in tokens]
    return RE_SPACES.sub(" ", " ".join(norm_tokens)).strip()


def fast_normalize_country(country: str) -> str:
    if not country:
        return ""
    c = str(country).strip().upper()
    if "US" in c or "UNITED STATES" in c or "USA" in c:
        return "US"
    elif "IN" in c or "INDIA" in c:
        return "IN"
    return c


def extract_blocking_keys_fast(name_norm: str, addr_norm: str, country_norm: str) -> list:
    """Baseline 4 Passes"""
    if not country_norm:
        return []

    keys = []

    # Pass 1: Exact Name
    if len(name_norm) >= 2:
        keys.append(("exact_name", country_norm, name_norm))

    # Pass 2: Name Tokens
    if name_norm:
        for tok in name_norm.split():
            if len(tok) >= 4 and tok not in NAME_STOP_WORDS:
                keys.append(("name_token", country_norm, tok))

    # Pass 3: Address Tokens
    if addr_norm:
        for tok in addr_norm.split():
            if len(tok) >= 4 and tok not in ADDRESS_STOP_WORDS:
                keys.append(("address_token", country_norm, tok))

    # Pass 4: Name Prefix
    if len(name_norm) >= 3:
        prefix = name_norm[:3].strip()
        if len(prefix) == 3:
            keys.append(("name_prefix", country_norm, prefix))

    return keys


def extract_blocking_keys_fast_tuned(name_norm: str, addr_norm: str, country_norm: str) -> list:
    """Tuned 7 Passes (Baseline 1-4 + Pass 5: char_ngram, Pass 6: postal_code, Pass 7: relaxed_token)"""
    if not country_norm:
        return []

    keys = []

    # Pass 1: Exact Name
    if len(name_norm) >= 2:
        keys.append(("exact_name", country_norm, name_norm))

    # Pass 2 & Pass 7: Name Tokens & Relaxed Tokens
    if name_norm:
        for tok in name_norm.split():
            if len(tok) >= 4 and tok not in NAME_STOP_WORDS:
                keys.append(("name_token", country_norm, tok))
            elif len(tok) == 3 and not tok.isdigit() and tok not in RELAXED_STOP_WORDS:
                keys.append(("relaxed_token", country_norm, tok))

    # Pass 3 & Pass 6: Address Tokens & Postal Codes
    if addr_norm:
        for tok in addr_norm.split():
            if len(tok) >= 4 and tok not in ADDRESS_STOP_WORDS:
                keys.append(("address_token", country_norm, tok))

        # Pass 6: Postal Code (5-6 digits)
        postalcodes = RE_POSTAL.findall(addr_norm)
        for pc in postalcodes:
            keys.append(("postal_code", country_norm, pc))

    # Pass 4: Name Prefix
    if len(name_norm) >= 3:
        prefix = name_norm[:3].strip()
        if len(prefix) == 3:
            keys.append(("name_prefix", country_norm, prefix))

    # Pass 5: Character N-Grams (3-grams)
    clean_name = name_norm.replace(" ", "")
    if len(clean_name) >= 4:
        ngrams = {clean_name[i:i+3] for i in range(len(clean_name) - 2)}
        for ng in ngrams:
            if ng not in COMMON_CHAR_NGRAMS and not ng.isdigit():
                keys.append(("char_ngram", country_norm, ng))

    return keys


def build_inverted_indexes(db_path: Path = INDEX_DB_PATH, tuned: bool = False):
    label_mode = "TUNED (7 PASSES)" if tuned else "BASELINE (4 PASSES)"
    print("=" * 80)
    print(f"APPROACH V2: BUILDING {label_mode} DISK-BACKED INVERTED INDEX")
    print("=" * 80)
    t0 = time.time()

    source_files = [
        ("Source 2", TRAIN_FILES["s2"]),
        ("Source 3", TRAIN_FILES["s3"])
    ]

    key_extractor = extract_blocking_keys_fast_tuned if tuned else extract_blocking_keys_fast

    # --- Phase 1: Fast Frequency Counting Pass ---
    print(f"\n[Phase 1/2] Streaming S2 & S3 to compute block key frequencies ({label_mode})...", flush=True)
    t1 = time.time()
    key_counts = Counter()
    total_processed = 0

    for label, filepath in source_files:
        if not filepath.exists():
            print(f"❌ Error: File not found {filepath}")
            continue

        print(f"  Counting frequencies for {label}...", flush=True)
        with open(filepath, mode="r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                total_processed += 1
                name_norm = fast_normalize_name(row.get("business_name"))
                addr_norm = fast_normalize_address(row.get("business_address"))
                country_norm = fast_normalize_country(row.get("country"))

                keys = key_extractor(name_norm, addr_norm, country_norm)
                for k in keys:
                    key_counts[k] += 1

    total_keys = len(key_counts)
    valid_keys = {k for k, count in key_counts.items() if count <= MAX_BLOCK_SIZE}
    pruned_keys_count = total_keys - len(valid_keys)

    print(f"  [DONE] Total Keys: {total_keys:,} | Valid Keys (<= {MAX_BLOCK_SIZE}): {len(valid_keys):,} | Pruned Keys (> {MAX_BLOCK_SIZE}): {pruned_keys_count:,} in {round(time.time() - t1, 2)}s", flush=True)

    del key_counts

    # --- Phase 2: Insertion of Valid Keys into SQLite ---
    print(f"\n[Phase 2/2] Populating SQLite inverted index ({db_path.name})...", flush=True)
    t2 = time.time()

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA journal_mode = MEMORY;")
    cur.execute("PRAGMA cache_size = 2000000;")
    cur.execute("PRAGMA temp_store = MEMORY;")

    cur.execute("""
        CREATE TABLE inverted_index (
            pass_id TEXT,
            country TEXT,
            block_key TEXT,
            entity_id TEXT
        );
    """)

    batch = []
    batch_size = 250_000
    total_inserted = 0

    for label, filepath in source_files:
        print(f"  Inserting valid index entries for {label}...", flush=True)
        with open(filepath, mode="r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                entity_id = row.get("entity_id", "").strip()
                name_norm = fast_normalize_name(row.get("business_name"))
                addr_norm = fast_normalize_address(row.get("business_address"))
                country_norm = fast_normalize_country(row.get("country"))

                keys = key_extractor(name_norm, addr_norm, country_norm)
                for pass_id, ctry, bkey in keys:
                    if (pass_id, ctry, bkey) in valid_keys:
                        batch.append((pass_id, ctry, bkey, entity_id))

                if len(batch) >= batch_size:
                    cur.executemany("INSERT INTO inverted_index VALUES (?, ?, ?, ?);", batch)
                    total_inserted += len(batch)
                    batch.clear()

    if batch:
        cur.executemany("INSERT INTO inverted_index VALUES (?, ?, ?, ?);", batch)
        total_inserted += len(batch)
        batch.clear()

    conn.commit()
    print(f"  [DONE] Inserted {total_inserted:,} valid index entries in {round(time.time() - t2, 2)}s", flush=True)

    # --- Phase 3: Create Composite Index for Fast Lookups ---
    print("\nCreating index idx_lookup on (pass_id, country, block_key)...", flush=True)
    t_idx = time.time()
    cur.execute("CREATE INDEX idx_lookup ON inverted_index(pass_id, country, block_key);")
    conn.commit()
    conn.close()
    print(f"  [DONE] B-Tree Index created in {round(time.time() - t_idx, 2)}s", flush=True)

    db_size_mb = round(db_path.stat().st_size / (1024 * 1024), 2)
    t_total = round(time.time() - t0, 2)

    print("\n" + "=" * 80)
    print(f"INVERTED INDEX BUILD COMPLETE ({label_mode})")
    print("=" * 80)
    print(f"Total Source Entities Processed : {total_processed:,}")
    print(f"Final Pruned Index Rows Stored  : {total_inserted:,}")
    print(f"Database File Size              : {db_size_mb} MB ({db_path.name})")
    print(f"Total Execution Time            : {t_total}s")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", action="store_true", help="Build tuned 7-pass index")
    args = parser.parse_args()

    if args.tuned:
        build_inverted_indexes(INDEX_TUNED_DB_PATH, tuned=True)
    else:
        build_inverted_indexes(INDEX_DB_PATH, tuned=False)
