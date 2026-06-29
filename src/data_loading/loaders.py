"""
Dataset loaders for EDGAR-CORPUS and S&P 500 earnings transcripts.
Loads from Hugging Face, takes small samples, saves as parquet.
"""

from pathlib import Path
import pandas as pd
from datasets import load_dataset


def load_edgar_sample(
    sample_size: int = 100,
    year: str = "2020",
    split: str = "train",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load a small sample of EDGAR-CORPUS filings.

    Args:
        sample_size: number of filings to sample
        year: which year's data to pull (EDGAR-CORPUS is split by year)
        split: 'train', 'validation', or 'test'
        seed: random seed for reproducibility

    Returns:
        pandas DataFrame with sampled filings
    """
    print(f"Loading EDGAR-CORPUS year={year}, split={split}...")
    ds = load_dataset(
        "eloukas/edgar-corpus",
        f"year_{year}",
        split=split,
    )
    print(f"Full dataset size: {len(ds)} filings")

    # Take a random sample for development
    sampled = ds.shuffle(seed=seed).select(range(min(sample_size, len(ds))))
    df = sampled.to_pandas()
    print(f"Sampled {len(df)} filings")
    return df


def load_earnings_sample(
    sample_size: int = 100,
    split: str = "train",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load a small sample of S&P 500 earnings call transcripts.

    Args:
        sample_size: number of transcripts to sample
        split: 'train', 'validation', or 'test' if available
        seed: random seed

    Returns:
        pandas DataFrame with sampled transcripts
    """
    print(f"Loading S&P 500 earnings transcripts split={split}...")
    ds = load_dataset(
        "glopardo/sp500-earnings-transcripts",
        split=split,
    )
    print(f"Full dataset size: {len(ds)} transcripts")

    sampled = ds.shuffle(seed=seed).select(range(min(sample_size, len(ds))))
    df = sampled.to_pandas()
    print(f"Sampled {len(df)} transcripts")
    return df


def save_sample(df: pd.DataFrame, output_path: Path) -> None:
    """Save a DataFrame to parquet, creating parent dirs if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


def inspect_dataframe(df: pd.DataFrame, name: str) -> dict:
    """
    Return a basic inspection summary of a DataFrame.
    Useful for writing inspection notes.
    """
    summary = {
        "name": name,
        "n_rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }

    # Identify text-like columns. Modern pandas/Arrow can label string columns
    # as 'object', 'string', 'str', 'string[python]', 'string[pyarrow]', or 'large_string'.
    TEXT_DTYPE_MARKERS = ("object", "string", "str", "large_string")

    text_lengths = {}
    for col in df.columns:
        dtype_str = str(df[col].dtype).lower()
        is_text_like = any(marker in dtype_str for marker in TEXT_DTYPE_MARKERS)

        if not is_text_like:
            continue

        try:
            series = df[col].dropna().astype(str)
            if len(series) == 0:
                continue
            lengths = series.str.len()
            text_lengths[col] = {
                "mean_chars": int(lengths.mean()),
                "min_chars": int(lengths.min()),
                "max_chars": int(lengths.max()),
                "median_chars": int(lengths.median()),
                "non_null_count": int(len(series)),
            }
        except Exception as e:
            text_lengths[col] = {"error": str(e)}

    summary["text_length_stats"] = text_lengths
    return summary