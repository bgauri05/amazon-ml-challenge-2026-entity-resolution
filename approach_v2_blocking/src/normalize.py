"""
Approach V2 Data Normalization Module
Provides streaming-reusable text normalization functions for business names, addresses, and countries.
Updated with targeted numeric token leading zero normalization for addresses.
"""

import sys
import re
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Legal Suffix Regex (including common Latin and Devanagari legal terms)
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

# Safe Address Abbreviation Mappings
ADDRESS_ABBR_MAP = {
    r"\broad\b": "rd",
    r"\bstreet\b": "st",
    r"\bavenue\b": "ave",
    r"\bdrive\b": "dr",
    r"\bboulevard\b": "blvd",
    r"\blane\b": "ln",
    r"\bsuite\b": "ste",
    r"\bapartment\b": "apt",
    r"\bbuilding\b": "bldg",
    r"\bexpressway\b": "expy",
    r"\bhighway\b": "hwy",
    r"\bparkway\b": "pkwy",
}


def remove_punctuation(text: str) -> str:
    """
    Replaces Unicode Punctuation (P) and Symbol (S) characters with spaces,
    preserving non-Latin letters, digits, and combining marks (vowels/diacritics).
    """
    return "".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in text)


def normalize_name(name: str) -> str:
    """
    Normalizes a business name:
    - Unicode NFKC normalization
    - Lowercase
    - '&' -> ' and '
    - Strips common legal suffixes (ltd, inc, llc, pvt, corp, company, etc.)
    - Replaces punctuation with space (preserving non-Latin letters & combining marks)
    - Normalizes whitespace
    """
    if not name:
        return ""

    # 1. Unicode Normalize (NFKC)
    text = unicodedata.normalize("NFKC", str(name))

    # 2. Lowercase
    text = text.lower()

    # 3. Ampersand replacement
    text = text.replace("&", " and ")

    # 4. Remove legal suffixes before removing punctuation
    text = LEGAL_SUFFIX_REGEX.sub(" ", text)

    # 5. Punctuation -> Space
    text = remove_punctuation(text)

    # 6. Second legal suffix cleanup pass after punctuation removal
    text = LEGAL_SUFFIX_REGEX.sub(" ", text)

    # 7. Whitespace normalization
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_address(address: str) -> str:
    """
    Normalizes a business address:
    - Unicode NFKC normalization
    - Lowercase
    - Replaces common address terms with standardized abbreviations (road -> rd, street -> st, etc.)
    - Replaces punctuation with space while preserving digits and numbers
    - Strips leading zeros from purely numeric tokens (e.g. 029569 -> 29569, 00014 -> 14)
    - Normalizes whitespace
    """
    if not address or not isinstance(address, str):
        return ""

    # 1. Unicode Normalize (NFKC)
    text = unicodedata.normalize("NFKC", address)

    # 2. Lowercase
    text = text.lower()

    # 3. Standardize common address abbreviations
    for pattern, replacement in ADDRESS_ABBR_MAP.items():
        text = re.sub(pattern, replacement, text)

    # 4. Punctuation -> Space
    text = remove_punctuation(text)

    # 5. Strip leading zeros from purely numeric address tokens
    tokens = text.split()
    norm_tokens = [str(int(t)) if (t.isdigit() and len(t) > 1) else t for t in tokens]

    # 6. Whitespace normalization
    text = re.sub(r"\s+", " ", " ".join(norm_tokens)).strip()

    return text


def normalize_country(country: str) -> str:
    """
    Normalizes country code/name to categorical 'US' or 'IN'.
    """
    if not country or not isinstance(country, str):
        return ""

    text = country.strip().upper()

    if "US" in text or "UNITED STATES" in text or "USA" in text:
        return "US"
    elif "IN" in text or "INDIA" in text:
        return "IN"

    return text


def normalize_row(row: dict) -> dict:
    """
    Streaming row helper: accepts a raw dictionary and returns a dictionary with
    both preserved original fields and normalized fields.
    """
    orig_name = str(row.get("business_name") or "")
    orig_addr = str(row.get("business_address") or "")
    orig_ctry = str(row.get("country") or "")

    return {
        "entity_id": row.get("entity_id", ""),
        "original_business_name": orig_name,
        "normalized_business_name": normalize_name(orig_name),
        "original_business_address": orig_addr,
        "normalized_business_address": normalize_address(orig_addr),
        "original_country": orig_ctry,
        "normalized_country": normalize_country(orig_ctry),
    }


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING APPROACH V2 DATA NORMALIZATION MODULE (WITH NUMERIC ADDRESS FIX)")
    print("=" * 80)

    test_samples = [
        {
            "label": "Leading Zero Numeric Address Test",
            "row": {
                "entity_id": "S3-211915957",
                "business_name": "Apex Pinnacle Group Music",
                "business_address": "029569 Forrest Rd, Albemarle, North Carolina 00014",
                "country": "US"
            }
        },
        {
            "label": "Indian / Devanagari Script Name & Address",
            "row": {
                "entity_id": "S2-002",
                "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
                "business_address": "KH NO. -00570/13, NEW DELHI, GULMOHAR COLONY, BHOPAL",
                "country": "IN"
            }
        }
    ]

    for idx, sample in enumerate(test_samples, 1):
        print(f"\n--- Sample {idx}: {sample['label']} ---")
        raw = sample["row"]
        norm = normalize_row(raw)

        print(f"Entity ID: {norm['entity_id']}")
        print(f"  Address [RAW]  : {norm['original_business_address']}")
        print(f"          [NORM] : {norm['normalized_business_address']}")

    print("\n" + "=" * 80)
    print("NORMALIZATION TEST COMPLETE")
    print("=" * 80)
