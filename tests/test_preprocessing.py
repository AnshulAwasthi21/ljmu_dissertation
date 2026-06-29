"""
Sanity tests for preprocessing functions.

The assert statement is the alarm mechanism. assert x == y means "if x equals y,
 do nothing; if they're not equal, crash loudly.
" So assert normalize_whitespace("  hello   world  ") == "hello world" 
reads as: "I claim that when I clean the messy string  hello   world , 
the function should return the tidy string hello world. Crash if it doesn't."

Note : Run with 'uv run python tests/test_preprocessing.py' from project root. 
If all tests print their checkmark and you see "All tests passed", you're good.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.cleaners import (
    normalize_whitespace,
    remove_control_chars,
    basic_clean,
    split_transcript_on_qa,
)


def test_normalize_whitespace():
    assert normalize_whitespace("  hello   world  ") == "hello world"
    # \n is a newline, \t is a tab. Both should be treated as whitespace and collapsed to single spaces.
    assert normalize_whitespace("a\n\nb\tc") == "a b c"
    assert normalize_whitespace("") == ""
    assert normalize_whitespace(None) == ""
    print("✓ normalize_whitespace")


def test_remove_control_chars():
    assert remove_control_chars("hello\x00world") == "helloworld"
    assert remove_control_chars("normal text") == "normal text"
    print("✓ remove_control_chars")


def test_basic_clean():
    dirty = "  Hello\x00\x01  \n\nworld  "
    assert basic_clean(dirty) == "Hello world"
    print("✓ basic_clean")


def test_split_transcript_on_qa():
    """
    Case 1 verifies the splitter finds the marker and divides correctly. 
    Case 2 verifies that when there's no marker, the function returns the whole text as prepared with was_split=False.
    """
    # Case 1: marker present
    text = "Opening remarks here. [Operator Instructions] First question..."
    prepared, qa, was_split = split_transcript_on_qa(text)
    assert was_split is True
    assert "Opening remarks" in prepared
    assert "[Operator Instructions]" in qa
    assert "First question" in qa

    # Case 2: marker absent
    text = "All prepared, no Q&A marker here."
    prepared, qa, was_split = split_transcript_on_qa(text)
    assert was_split is False
    assert prepared == text
    assert qa == ""
    print("✓ split_transcript_on_qa")

def test_strip_participant_header():
    """
    Three scenarios: a real participant header (should be stripped), 
    a plain operator opening with no header (should pass through), 
    and a named-speaker opening like David Niederman (also no header, also passes through).
    """
    from src.preprocessing.cleaners import strip_participant_header
    
    # Real format from your data
    # BEFORE
    # text = "Executives: Tim Cook - CEO Luca Maestri - CFO Analysts: Toni Sacconaghi - Bernstein Operator : Good morning, welcome to the call..."
    # cleaned, stripped = strip_participant_header(text)
    # assert stripped is True
    # assert cleaned.startswith("Operator")
    # assert "Tim Cook" not in cleaned

    # AFTER
    text = "Executives: Tim Cook - CEO Luca Maestri - CFO Analysts: Toni Sacconaghi - Bernstein. Operator : Good morning, welcome to the call..."
    cleaned, stripped = strip_participant_header(text)
    assert stripped is True
    assert cleaned.lstrip(". ").startswith("Operator")   # tightened marker returns from the boundary char
    assert "Tim Cook" not in cleaned
    
    # Operator-led transcript (no header) — should be unchanged
    text = "Operator : Good morning, welcome..."
    cleaned, stripped = strip_participant_header(text)
    assert stripped is False
    assert cleaned == text
    
    # Direct opening (Transcript 24/25 style) — should be unchanged
    text = "David Niederman : Good afternoon, and thank you for joining us..."
    cleaned, stripped = strip_participant_header(text)
    assert stripped is False
    assert cleaned == text
    
    print("✓ strip_participant_header")


def test_split_transcript_on_qa_skips_early_marker():
    """
    When [Operator Instructions] appears twice - once in the operator's
    opening (early), once at the real Q&A boundary (later) - the splitter
    must pick the later occurrence. The 2000-char min_prepared_chars
    threshold skips the early procedural mention.

    Note : The "CEO speaking. " * 1000 trick is just a quick way to manufacture a long block of text
    it creates a string with "CEO speaking. " repeated 1000 times, which gives you about 14,000 characters. 
    Exact length doesn't matter; it just needs to be long enough that the second marker lands past the 2000-char threshold.
    """
    # Build a synthetic transcript with two markers:
    #   - First at position ~100 (inside operator's opening)
    #   - Second at position ~15000 (real Q&A boundary)
    operator_opening = (
        "Operator : Good morning, welcome to the Q1 2020 earnings call. "
        "[Operator Instructions] I would now like to hand the call over "
        "to the CEO. "
    )
    prepared_remarks_body = "Thank you operator. " + ("CEO speaking. " * 1000)
    qa_section = "[Operator Instructions] We will now take your questions. Analyst : My question is..."
    
    text = operator_opening + prepared_remarks_body + qa_section

    prepared, qa, was_split = split_transcript_on_qa(text)

    # The split should have happened (marker is present)
    assert was_split is True, "Splitter should detect the marker"

    # The prepared section should contain the real CEO content,
    # NOT just the operator's 60-char opening
    assert len(prepared) > 2000, (
        f"Expected prepared > 2000 chars (real remarks), got {len(prepared)}"
    )
    assert "CEO speaking" in prepared, "Real prepared remarks should be in prepared"

    # The qa section should start at the SECOND marker, not the first
    assert qa.startswith("[Operator Instructions] We will now take your questions"), (
        "qa should start at the late marker, not the early one"
    )

    print("✓ split_transcript_on_qa skips early marker")


# Adding this call to the __main__ block at the bottom


if __name__ == "__main__":
    test_normalize_whitespace()
    test_remove_control_chars()
    test_basic_clean()
    test_split_transcript_on_qa()
    test_split_transcript_on_qa_skips_early_marker()  # new
    test_strip_participant_header()
    print("\nAll tests passed.")