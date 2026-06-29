# Beyond the Prompt : Comparative Vectorisation & Retrieval for Financial Summarisation

Code for the LJMU MSc dissertation *"Beyond the Prompt: A Comparative Study of
Vectorisation and Retrieval Techniques for Improving Generative AI Summarisation
Accuracy Across Formal and Conversational Financial Narratives."*

The study runs a controlled, single-variable comparison of three retrieval
strategies - sparse (BM25), dense (`bge-small-en-v1.5` with a FAISS exact-cosine
index), and a hybrid that fuses them through Reciprocal Rank Fusion - over two
financial corpora: formal EDGAR 10-K filings and conversational S&P 500
earnings-call transcripts. The chunk store, golden query set, generator, and
prompt are all held constant, so that only the retriever varies and any
difference in the result is attributable to retrieval alone.

## Status

> **This repository is shared during the evaluation / grading phase.**
> Study **results, raw data, and dissertation notes are withheld** until grading
> is complete. Each withheld directory keeps a short placeholder note and will be
> populated after final submission.


<h3>Folder structure should look like:</h3>
<pre>
code/
├── data/
│   ├── raw/               ✓ (downloaded HF datasets)
│   ├── processed/         ✓ (cleaned, chunked)
│   └── samples/           ✓ (small subsets for dev)
├── notebooks/             ✓ (exploration + phase notebooks)
│   ├── 01_load_and_inspect_datasets.ipynb
│   ├── 02_clean_and_prepare_samples.ipynb
│   ├── 03_chunk_documents
│   ├── 04_bm25_baseline.ipynb
│   ├── 05_dense_retrieval.ipynb
│   ├── 06_hybrid_rrf.ipynb
│   ├── 07_evaluation.ipynb
│   ├── 08_summarisation.ipynb
│   └── 09_results_tables_and_charts.ipynb
├── src/
│   ├── data_loading/      ✓
│   ├── preprocessing/     ✓
│   ├── chunking/          ✓
│   ├── retrieval/         ✓
│   ├── generation/        ✓
│   └── evaluation/        ✓
├── experiments/           ✓ (results CSVs, run logs)
├── outputs/
│   ├── retrieval_results/ ✓
│   ├── summaries/         ✓
│   └── evaluation/        ✓
├── reports/               ✓ (figures, tables for thesis)
├── configs/               ✓ (YAML config files)
├── tests/                 ← ADD: small test scripts to verify modules work
├── logs/                  ← ADD: experiment run logs
├── .gitignore             ← ADD: exclude data/, .venv/, __pycache__
├── pyproject.toml         ← ADD: uv project file (Phase 1 will create)
├── README.md              ← ADD: brief project description
└── notes/                 ← ADD: your experiment notes (markdown files per phase)
</pre>

<h3>Full Phase Roadmap (We'll Execute One at a Time)
<br>
Here's the sequence so you see the path.</h3>
<pre>

Phase   Notebook/Module                 Goal
1       01_load_and_inspect_datasets    Load HF datasets, inspect schema, save 100-doc samples as parquet
2       02_clean_and_prepare_samples    Strip noise, normalize text, handle tables/speaker turns
3       03_chunk_documents              Apply chunking strategy, save chunked corpus
4       04_bm25_baseline                Build BM25 index, run retrieval, measure Precision@K, Recall@K, MRR
5       05_dense_retrieval              Embed chunks with sentence-transformers, FAISS index, same metrics
6       06_hybrid_rrf                   Combine sparse + dense via RRF, same metrics
7       07_summarisation                Fixed LLM + fixed prompt, generate summaries from each retriever's context
8       08_evaluation                   ROUGE-L, LLM-as-judge faithfulness, numerical fact-check, HITL spreadsheet
9       09_results_tables_and_charts    Aggregate everything into thesis-ready tables and plots
</pre>

## What's included

| Path | Contents |
|------|----------|
| `src/` | Importable pipeline modules: data loading, preprocessing, chunking, retrieval (BM25 / dense / hybrid), generation, and evaluation |
| `tests/` | Offline unit tests mirroring the modules |
| `notebooks/` | Top-level analysis notebooks, one per phase (data-prep and helper *queries* notebooks are not published) |
| `configs/` | Configuration files (reserved) |
| `pyproject.toml`, `uv.lock` | Pinned, reproducible environment |

## Withheld until after grading

`data/`, `experiments/`, `outputs/`, `reports/`, `logs/`, `notes/`, plus the
data-prep and *queries* helper notebooks.

## Environment

Python 3.13, managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The datasets are public on Hugging Face (EDGAR-CORPUS; an S&P 500 earnings-call
transcript set) and are downloaded into `data/` at run time. They are not
committed to this repository.

## Reproducing the pipeline

The notebooks are numbered by phase and import logic from `src/` rather than
re-implementing it. The intended order is data loading and inspection →
cleaning → chunking → golden evaluation set → the three retrievers → generation
→ evaluation, with an optional larger-corpus robustness run.

## Licence / academic note

This code accompanies an MSc dissertation submitted to Liverpool John Moores
University. Please do not reuse it for assessed work.
