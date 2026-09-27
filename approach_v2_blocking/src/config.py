"""
Configuration & Path Management for Approach V2 Blocking
"""

import os
from pathlib import Path

# Path Hierarchy
SRC_DIR = Path(__file__).resolve().parent
APPROACH_ROOT = SRC_DIR.parent
PROJECT_ROOT = APPROACH_ROOT.parent

# Raw Data Directories
RAW_DATA_DIR = PROJECT_ROOT / "dataset"
if not RAW_DATA_DIR.exists():
    RAW_DATA_DIR = PROJECT_ROOT / "student_resource" / "dataset"

TRAIN_DIR = RAW_DATA_DIR / "train"
TEST_DIR = RAW_DATA_DIR / "test"

# Approach V2 Specific Directories
DATA_DIR = APPROACH_ROOT / "data"
INDEXES_DIR = APPROACH_ROOT / "indexes"
CANDIDATES_DIR = APPROACH_ROOT / "candidates"
FEATURES_DIR = APPROACH_ROOT / "features"
MODELS_DIR = APPROACH_ROOT / "models"
OUTPUTS_DIR = APPROACH_ROOT / "outputs"
EXPERIMENTS_DIR = APPROACH_ROOT / "experiments"

# Ensure directories exist
for d in [DATA_DIR, INDEXES_DIR, CANDIDATES_DIR, FEATURES_DIR, MODELS_DIR, OUTPUTS_DIR, EXPERIMENTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset TSV Files
TRAIN_FILES = {
    "s1": TRAIN_DIR / "train_source1.tsv",
    "s2": TRAIN_DIR / "train_source2.tsv",
    "s3": TRAIN_DIR / "train_source3.tsv",
    "gt": TRAIN_DIR / "train_ground_truth.tsv"
}

TEST_FILES = {
    "s1": TEST_DIR / "test_source1.tsv",
    "s2": TEST_DIR / "test_source2.tsv",
    "s3": TEST_DIR / "test_source3.tsv"
}

# Index Database Paths
INDEX_DB_PATH = INDEXES_DIR / "inverted_index.db"
INDEX_TUNED_DB_PATH = INDEXES_DIR / "inverted_index_tuned.db"

# Candidate Pair Files inside candidates/
CANDIDATE_FILES = {
    "test_1k_candidates": CANDIDATES_DIR / "test_1k_candidates.parquet",
    "test_1k_candidates_tuned": CANDIDATES_DIR / "test_1k_candidates_tuned.parquet",
    "test_1k_candidates_ranked": CANDIDATES_DIR / "test_1k_candidates_ranked.parquet",
    "test_1k_candidates_ranked_num": CANDIDATES_DIR / "test_1k_candidates_ranked_num.parquet",
    "train_candidates": CANDIDATES_DIR / "train_candidates.parquet",
    "test_candidates": CANDIDATES_DIR / "test_candidates.parquet"
}

# Multi-Pass Blocking Configuration
MAX_BLOCK_SIZE = 500
MAX_CANDIDATES_PER_ENTITY = 50
MAX_CANDIDATES_PER_ENTITY_TUNED = 100

# Pre-Ranking Relevance Weights
RELEVANCE_WEIGHTS = {
    "name_jaccard_weight": 4.0,
    "address_jaccard_weight": 3.0,
    "exact_name_bonus": 5.0,
    "postal_match_bonus": 2.0,
    "numeric_overlap_bonus": 2.0,
    "multi_pass_bonus": 1.5,
}

# Generic Stop Words for Name Token Blocking
NAME_STOP_WORDS = {
    "inc", "ltd", "llc", "corp", "co", "the", "and", "company", "limited", "corporation",
    "pvt", "private", "gmbh", "group", "services", "solutions", "global", "international",
    "enterprises", "enterprise", "holding", "holdings", "technologies", "tech", "systems",
    "system", "trading", "traders", "logistics", "associates", "management", "consulting",
    "works", "store", "shop", "center", "centre", "market", "marketing", "india", "usa",
    "industries", "industry", "supply", "supplies", "auto", "motors", "care", "home", "food",
    "foods", "hotel", "resort", "mart", "supermarket"
}

# Address Stop Words for Address Token Blocking
ADDRESS_STOP_WORDS = {
    "street", "st", "road", "rd", "avenue", "ave", "drive", "dr", "lane", "ln", "boulevard", "blvd",
    "suite", "ste", "apartment", "apt", "building", "bldg", "floor", "fl", "unit", "plot", "house",
    "hwy", "highway", "pkwy", "parkway", "cross", "main", "near", "opposite", "opp", "behind",
    "next", "sector", "block", "nagar", "colony", "dist", "district", "state", "city", "west",
    "east", "north", "south", "delhi", "new", "bhopal", "mumbai", "chennai", "kolkata", "bangalore",
    "hyderabad", "pune", "ahmedabad", "jaipur", "gurgaon", "noida", "post", "box", "po", "zip",
    "code", "pin", "area", "bazar", "bazaar"
}

# Common N-Grams to Exclude in Pass 5
COMMON_CHAR_NGRAMS = {
    "the", "and", "inc", "ltd", "pvt", "cor", "orp", "com", "pan", "any", "lim", "ite", "ted",
    "ser", "vic", "ces", "sol", "uti", "ion", "gro", "oup", "ent", "ter", "pri", "val", "ate",
    "int", "ern", "nat", "tio", "nal", "tech", "sys", "tem", "tra", "din", "log", "ist", "ics"
}

# Short 3-Character Stop Tokens for Pass 7 Relaxed Token Blocking
RELAXED_STOP_WORDS = {
    "inc", "ltd", "pvt", "corp", "corp.", "co.", "the", "and", "for", "out", "new", "top", "all",
    "one", "two", "six", "ten", "cat", "dog", "lab", "hub", "law", "tax", "max", "bar", "pub",
    "spa", "gym", "net", "box", "off", "via", "del", "san", "los", "las", "st.", "rd.", "ave"
}
