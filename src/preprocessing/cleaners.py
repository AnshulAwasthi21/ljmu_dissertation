"""
Text cleaning and document preparation for EDGAR filings and earnings transcripts.

Design principles:
- Light cleaning only. Heavy cleaning introduces methodological confounds.
- Each function is small and testable.
- Output schema is unified across both corpora.
"""

import re
from typing import Optional
import pandas as pd


# -------------------------------------------------------------------
# Generic text cleaning utilities
# -------------------------------------------------------------------

def normalize_whitespace(text: str, preserve_newlines: bool = False) -> str:
    """
    Collapse whitespace runs. If preserve_newlines=True, newlines are kept
    as line boundaries (collapsing only spaces/tabs within lines and
    deduplicating consecutive blank lines). Used for transcripts where
    speaker turns need to be detectable downstream.
    """
    if not isinstance(text, str):
        return ""
    if preserve_newlines:
        # Collapse spaces/tabs within each line, strip per-line
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
        # Drop empty lines but keep newlines as separators between content
        non_empty = [ln for ln in lines if ln]
        return "\n".join(non_empty)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_control_chars(text: str) -> str:
    """
    Strip non-printable control characters and Unicode formatting marks.
    
    Removes:
    - C0 control chars (\\x00-\\x1f) except common whitespace
    - Byte Order Mark (\\ufeff) — appears at start of ~66% of earnings transcripts
    - Zero-width characters (\\u200b-\\u200f, \\u2028-\\u202f) that pollute embeddings
    """
    if not isinstance(text, str):
        return ""
    # Drop C0 control chars (keep \t, \n, \r — they get normalized later)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)
    # Drop BOM and zero-width / formatting marks
    text = re.sub(r"[\ufeff\u200b-\u200f\u2028-\u202f]", "", text)
    return text


def basic_clean(text: str, preserve_newlines: bool = False) -> str:
    """Apply minimal cleaning suitable for both corpora."""
    text = remove_control_chars(text)
    text = normalize_whitespace(text, preserve_newlines=preserve_newlines)
    return text

# -------------------------------------------------------------------
# Earnings transcript header stripping
# -------------------------------------------------------------------

# Pattern: transcript starts with "Executives:" block (32/100 in our sample)
# These blocks list executives and analysts before the actual call begins.
# We strip them by finding the first speaker turn that looks like the call opening.
PARTICIPANT_HEADER_PATTERN = re.compile(
    r"^\s*Executives?\s*:", re.IGNORECASE
)

# The actual call typically begins with "Operator :" after the header.
# We use this as the marker to find where the real content starts.
# CALL_START_MARKER = re.compile(r"\bOperator\s*:", re.IGNORECASE)
# CALL_START_MARKER = re.compile(r"(?:^|\n|\.\s)Operator\s*:", re.IGNORECASE)

# (?:^|\n|\.\s) - match one of three things: start of the entire string (^), or a newline (\n), 
# or a period-followed-by-whitespace (\.\s). 
# The (?:...) is a "non-capturing group" - it groups the alternatives without saving the match.
# Operator - literal text.
# \s*: - zero or more whitespace, then a colon.
# \b (word boundary)

CALL_START_MARKER = re.compile(r"(?:^|\n|\.\s)Operator\b\s*:", re.IGNORECASE)


def strip_participant_header(text: str) -> tuple[str, bool]:
    """
    If a transcript starts with an 'Executives:' participant list block,
    strip it and return the text starting from the first 'Operator :' marker.
    
    Returns (cleaned_text, was_stripped).
    
    If no header is detected, returns (text, False) unchanged.
    """
    if not isinstance(text, str) or not text.strip():
        return text, False
    
    # Check if this transcript starts with a participant header
    if not PARTICIPANT_HEADER_PATTERN.match(text):
        return text, False
    
    # Find the first 'Operator :' marker, which signals the call's actual start
    match = CALL_START_MARKER.search(text)
    if match is None:
        # Header detected but no Operator marker found — leave as-is and flag
        return text, False
    
    # Return text from the Operator marker onward
    return text[match.start():], True

# -------------------------------------------------------------------
# EDGAR-specific preparation
# -------------------------------------------------------------------

# Sections we care about, based on Phase 1 inspection
EDGAR_TARGET_SECTIONS = ["section_1", "section_1A", "section_7", "section_8"]
EDGAR_MIN_SECTION_CHARS = 500  # below this we treat the section as empty


def prepare_edgar_documents(
    df: pd.DataFrame,
    target_sections: Optional[list] = None,
    min_chars: int = EDGAR_MIN_SECTION_CHARS,
) -> pd.DataFrame:
    """
    Convert wide EDGAR DataFrame (one row per filing, many section columns)
    into a long DataFrame (one row per filing-section).

    Returns DataFrame with unified schema:
        doc_id, source, corpus_type, subtype, text, char_length, metadata
    """
    if target_sections is None:
        target_sections = EDGAR_TARGET_SECTIONS

    rows = []
    for filing_idx, row in df.iterrows():
        filename = row.get("filename", f"unknown_{filing_idx}")
        cik = row.get("cik", "")
        year = row.get("year", "")

        for section in target_sections:
            if section not in row:
                continue
            raw_text = row[section]
            cleaned = basic_clean(str(raw_text)) if raw_text is not None else ""

            # Skip empty/stub sections
            if len(cleaned) < min_chars:
                continue

            doc_id = f"edgar_{filing_idx:04d}_{section}"
            rows.append({
                "doc_id": doc_id,
                "source": "edgar",
                "corpus_type": "formal",
                "subtype": section,
                "text": cleaned,
                "char_length": len(cleaned),
                "metadata": {
                    "filename": filename,
                    "cik": cik,
                    "year": year,
                    "filing_idx": int(filing_idx),
                },
            })

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Earnings transcript preparation
# -------------------------------------------------------------------

# Boundary marker confirmed in Phase 1 (77/100 transcripts contain it)
QA_BOUNDARY_MARKER = "[Operator Instructions]"


def split_transcript_on_qa(text: str, min_prepared_chars: int = 2000,) -> tuple[str, str, bool]:
    """
    Split a transcript into (prepared_remarks, qa_section, was_split).

    If the boundary marker is found, returns (text_before, text_after, True).
    If not found, returns (text, "", False) — caller should treat as 'full'.

    The first occurrence of [Operator Instructions] is treated as the boundary.
    Earlier occurrences (rare) would be in the operator's opening, but in
    practice the marker reliably indicates the Q&A start.
    """
    positions = []
    start = 0
    while True:
        idx = text.find(QA_BOUNDARY_MARKER, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + len(QA_BOUNDARY_MARKER)

    if not positions:
        return text, "", False

    # Return first occurrence with enough preceding content
    for idx in positions:
        if idx >= min_prepared_chars:
            prepared = text[:idx]
            qa = text[idx:]
            return prepared, qa, True

    # All occurrences are early — last is best guess for QA boundary
    idx = positions[-1]
    prepared = text[:idx]
    qa = text[idx:]
    return prepared, qa, True


def prepare_earnings_documents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert earnings DataFrame (one row per call) into a long DataFrame
    where each call is split into prepared_remarks and qa parts.

    Calls without a detectable Q&A boundary become a single 'full' unit.
    Calls where the boundary was detected but the prepared section is below
    threshold (a "degenerate split" — e.g. an early operator-opening marker
    with no substantive prepared remarks before it) are ALSO treated as
    'full', because in that case the qa side actually contains the whole
    call and labeling it "qa" would mislabel scripted prepared remarks as
    spontaneous Q&A.

    Participant headers (e.g., 'Executives: ...') are stripped before splitting.
    """
    rows = []

    dropped_too_short = 0  # counter for transparency (reproducibility note)

    for call_idx, row in df.iterrows():
        raw = row.get("transcript", "")
        if not isinstance(raw, str) or len(raw) < 1000:
            dropped_too_short += 1
            continue

        cleaned = basic_clean(raw, preserve_newlines=True)

        # Strip participant header if present (32% of transcripts)
        cleaned, header_stripped = strip_participant_header(cleaned)

        prepared, qa, was_split = split_transcript_on_qa(cleaned)

        # A split is only "genuine" if the prepared section is substantive.
        # If prepared falls below 500 chars, the marker landed too early
        # (operator opening, not Q&A transition), so qa holds the whole call.
        genuine_split = was_split and len(prepared) >= 500

        meta = {
            "ticker": row.get("ticker", ""),
            "company": row.get("company", ""),
            "sector": row.get("sector", ""),
            "industry": row.get("industry", ""),
            "year": row.get("year", None),
            "quarter": row.get("quarter", None),
            "earnings_date": row.get("earnings_date", ""),
            "call_idx": int(call_idx),
            "was_split": was_split,           # keep: marker was detected at all
            "genuine_split": genuine_split,   # new: split produced real prepared
            # True only for calls where a marker was found but the prepared
            # side was too short — these get relabeled "full" for traceability
            "degenerate_split": was_split and not genuine_split,
            "header_stripped": header_stripped,
        }

        if genuine_split:
            rows.append({
                "doc_id": f"earnings_{call_idx:04d}_prepared",
                "source": "earnings",
                "corpus_type": "conversational",
                "subtype": "prepared_remarks",
                "text": prepared,
                "char_length": len(prepared),
                "metadata": meta,
            })
            rows.append({
                "doc_id": f"earnings_{call_idx:04d}_qa",
                "source": "earnings",
                "corpus_type": "conversational",
                "subtype": "qa",
                "text": qa,
                "char_length": len(qa),
                "metadata": meta,
            })
        else:
            # Covers two cases: no marker at all (was_split=False), and
            # degenerate splits (was_split=True but prepared < 500).
            rows.append({
                "doc_id": f"earnings_{call_idx:04d}_full",
                "source": "earnings",
                "corpus_type": "conversational",
                "subtype": "full",
                "text": cleaned,
                "char_length": len(cleaned),
                "metadata": meta,
            })

    if dropped_too_short > 0:
        print(f"prepare_earnings_documents: dropped {dropped_too_short} "
              f"transcripts shorter than 1000 chars (likely empty or stub rows)")

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Combined corpus utility
# -------------------------------------------------------------------

def combine_corpora(edgar_df: pd.DataFrame, earnings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate prepared EDGAR and earnings DataFrames into a single
    corpus DataFrame for downstream retrieval.
    """
    combined = pd.concat([edgar_df, earnings_df], ignore_index=True)
    return combined