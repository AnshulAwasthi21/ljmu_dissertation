"""
tests/test_chunking.py

Tests for src/chunking/chunkers.py.

The single most important test here is `test_flat_transcript_without_newlines_
splits_into_many_turns`: it locks in the Phase-3 design decision that earnings
speaker turns are detected by the " : " label convention and NOT by newlines,
because the source transcripts are a single flat string (Phase 2 log #10).
If a future edit reintroduces newline-anchored detection, that test fails.

Run with:  pytest tests/test_chunking.py -v
or 
Run with:  python -m pytest tests/ -v # which will run every test_*.py under tests dir

The python -m form puts your current directory on sys.path, so src is importable. 
(Plain pytest usually works too thanks to rootdir detection, but python -m pytest is the robust habit.)

"""

import json

import pandas as pd
import pytest

from src.chunking.chunkers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    chunk_by_speaker_turns,
    chunk_corpus,
    recursive_character_split,
    split_into_speaker_turns,
    strategy_for_source,
)

# ----------------------------------------------------------------------
# Fixtures: realistic, newline-free text shaped like the real corpus
# ----------------------------------------------------------------------
FLAT_TRANSCRIPT = (
    "Operator : Good day, and welcome to the Second Quarter Earnings Call. "
    "[Operator Instructions] I would now like to hand the call over to Mike. "
    "Michael Petters : Thanks. Good morning, everyone. Sales of $1.7 billion "
    "were down 3% from last year. "
    "Gautam Khanna : I was wondering if you could quantify the cash "
    "contribution to the pension for next year? "
    "Christopher Kastner : It is really too early for next year. We will "
    "assess that as we move through the year. "
    "Michael Petters : Our objective is to get the ship out as fast as we can."
)

# A long EDGAR-style flat prose block (no newlines, like the cleaned filings).
EDGAR_PROSE = (
    "The Company designs, manufactures and sells a broad range of products. "
    "Revenue increased 12% year over year driven by demand in core segments. "
) * 60


def test_short_text_returns_single_chunk_unchanged():
    short = "The Company reported revenue of $5 million this quarter."
    chunks = recursive_character_split(short, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
    assert chunks == [short]


def test_recursive_never_exceeds_chunk_size():
    chunks = recursive_character_split(EDGAR_PROSE, 1500, 200)
    assert len(chunks) > 1
    assert all(len(c) <= 1500 for c in chunks)


def test_recursive_has_overlap_between_consecutive_chunks():
    # With sentence-granular pieces shorter than the overlap budget, adjacent
    # chunks should share trailing/leading content.
    chunks = recursive_character_split(EDGAR_PROSE, 1500, 300)
    shared = 0
    for a, b in zip(chunks, chunks[1:]):
        tail = a[-60:]
        if tail and tail in b:
            shared += 1
    assert shared >= 1, "expected at least one overlapping boundary"


def test_flat_transcript_without_newlines_splits_into_many_turns():
    # THE regression test: this string contains zero newlines.
    assert "\n" not in FLAT_TRANSCRIPT
    turns = split_into_speaker_turns(FLAT_TRANSCRIPT)
    assert len(turns) >= 4, (
        "speaker turns must be detected by the ' : ' label convention; a "
        "newline-anchored approach would return a single blob here"
    )
    # Every detected turn begins with a capitalised speaker label.
    assert turns[0].startswith("Operator :")
    assert any(t.startswith("Gautam Khanna :") for t in turns)


def test_speaker_boundary_does_not_absorb_previous_sentence_period():
    turns = split_into_speaker_turns(FLAT_TRANSCRIPT)
    # The operator's closing "... over to Mike." must stay in the operator turn,
    # not get pulled into the "Michael Petters :" label.
    assert any(t.startswith("Michael Petters :") for t in turns)
    assert not any(t.startswith("Mike. Michael") for t in turns)


def test_digits_and_brackets_are_not_mistaken_for_speakers():
    turns = split_into_speaker_turns(FLAT_TRANSCRIPT)
    # "[Operator Instructions]" and "Q2"/numeric tokens must not create turns.
    assert not any(t.startswith("Q2 :") for t in turns)
    assert not any(t.lstrip().startswith("[Operator Instructions] :") for t in turns)


def test_speaker_packing_never_exceeds_chunk_size():
    chunks = chunk_by_speaker_turns(FLAT_TRANSCRIPT, 1500, 200)
    assert len(chunks) >= 1
    assert all(len(c) <= 1500 for c in chunks)


def test_oversized_single_turn_is_subsplit():
    # One enormous answer turn (far bigger than chunk_size) must be broken up.
    big_answer = "word " * 1000  # ~5000 chars, single turn
    transcript = "Operator : Question please. Jane Smith : " + big_answer
    chunks = chunk_by_speaker_turns(transcript, 1500, 200)
    assert len(chunks) > 1
    assert all(len(c) <= 1500 for c in chunks)


def test_no_empty_chunks_emitted():
    for chunks in (
        recursive_character_split(EDGAR_PROSE, 1500, 200),
        chunk_by_speaker_turns(FLAT_TRANSCRIPT, 1500, 200),
    ):
        assert all(c.strip() for c in chunks)


def test_strategy_routing():
    assert strategy_for_source("earnings") == "speaker_turn"
    assert strategy_for_source("edgar") == "recursive_char"
    assert strategy_for_source("anything_else") == "recursive_char"


def _toy_corpus():
    return pd.DataFrame([
        {
            "doc_id": "edgar_0001_section_7", "source": "edgar",
            "corpus_type": "formal", "subtype": "section_7",
            "text": EDGAR_PROSE, "char_length": len(EDGAR_PROSE),
            "metadata": {"cik": "123", "year": "2020"},
        },
        {
            "doc_id": "earnings_0000_qa", "source": "earnings",
            "corpus_type": "conversational", "subtype": "qa",
            "text": FLAT_TRANSCRIPT, "char_length": len(FLAT_TRANSCRIPT),
            "metadata": {"ticker": "HII", "sector": "Industrials"},
        },
    ])


def test_chunk_corpus_inherits_ids_and_metadata():
    out = chunk_corpus(_toy_corpus(), 1500, 200)
    # Every chunk_id begins with its parent doc_id -> chunks cannot merge docs.
    assert all(r.chunk_id.startswith(r.doc_id) for r in out.itertuples())
    # Inherited fields match parent.
    edgar_rows = out[out.doc_id == "edgar_0001_section_7"]
    assert (edgar_rows.subtype == "section_7").all()
    assert edgar_rows.iloc[0]["metadata"] == {"cik": "123", "year": "2020"}
    earn_rows = out[out.doc_id == "earnings_0000_qa"]
    assert (earn_rows.corpus_type == "conversational").all()


def test_chunk_index_is_contiguous_and_n_chunks_correct():
    out = chunk_corpus(_toy_corpus(), 1500, 200)
    for doc_id, grp in out.groupby("doc_id"):
        idx = sorted(grp["chunk_index"].tolist())
        assert idx == list(range(len(grp)))
        assert (grp["n_chunks"] == len(grp)).all()


def test_metadata_string_is_deserialised():
    df = _toy_corpus()
    df.loc[0, "metadata"] = json.dumps({"cik": "999"})  # simulate parquet load
    out = chunk_corpus(df, 1500, 200)
    edgar_meta = out[out.doc_id == "edgar_0001_section_7"].iloc[0]["metadata"]
    assert edgar_meta == {"cik": "999"}


def test_every_parent_doc_has_at_least_one_chunk():
    corpus = _toy_corpus()
    out = chunk_corpus(corpus, 1500, 200)
    assert set(out["doc_id"]) == set(corpus["doc_id"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
