"""
preprocess.py — Data loading and cleaning for "The Shape of News" project.

Encapsulates the pipeline applied in main_notebook.ipynb:
    download MIND → load → audit → normalize → lemmatize → dedupe → save

Two ways to use this:

1. From the command line (produces a clean parquet file):
       python src/preprocess.py --output data/news_clean.parquet

2. From a notebook (returns the cleaned DataFrame in memory):
       from src.preprocess import run_pipeline
       news = run_pipeline()

Author: Anika Garg
Course: Data Mining & Analysis Final Project
"""

from __future__ import annotations

import argparse
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

MIND_URL = (
    "https://huggingface.co/datasets/yjw1029/MIND/"
    "resolve/main/MINDlarge_train.zip"
)

NEWS_COLUMNS = [
    "id", "category", "subcategory", "title", "abstract",
    "url", "entities_title", "entities_abstract",
]

# News-specific stopwords on top of standard English. These dominate news
# TF-IDF without distinguishing categories — reporting verbs, weekdays/months,
# generic filler, wire-service tokens.
NEWS_STOPWORDS = {
    # reporting verbs
    "said", "says", "told", "asked", "added", "noted", "reported", "reports",
    "according", "stated", "announced", "confirmed", "explained",
    "call", "calls", "called",
    # temporal
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "today", "yesterday", "tomorrow", "week", "weeks", "year", "years",
    "month", "months", "day", "days", "morning", "evening", "night",
    # generic filler
    "new", "news", "first", "last", "also", "people", "time", "times", "way",
    "thing", "things", "get", "got", "going", "make", "made", "take", "took",
    "see", "know", "think", "want", "really", "much", "many", "one", "two", "three",
    # wire/source/media artifacts
    "reuters", "ap", "cnn", "bbc", "photo", "getty", "image", "images",
    "video", "click", "read", "share", "tweet", "article", "story",
}


# ─────────────────────────────────────────────────────────────────────
# Stage 1: Download & Load
# ─────────────────────────────────────────────────────────────────────

def download_mind(target_dir: str = "MINDlarge_train") -> Path:
    """Download and extract the MIND large training set if not already present."""
    target = Path(target_dir)
    if target.exists():
        print(f"[download] {target_dir} already exists, skipping")
        return target

    zip_path = Path("MINDlarge_train.zip")
    print(f"[download] fetching {MIND_URL}")
    urllib.request.urlretrieve(MIND_URL, zip_path)

    print(f"[download] extracting to {target_dir}/")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall()

    return target


def load_news(data_dir: str = "MINDlarge_train") -> pd.DataFrame:
    """Read news.tsv into a DataFrame with named columns."""
    path = Path(data_dir) / "news.tsv"
    news = pd.read_csv(path, sep="\t", header=None)
    news.columns = NEWS_COLUMNS
    print(f"[load] {len(news):,} articles from {path}")
    return news


# ─────────────────────────────────────────────────────────────────────
# Stage 2: Audit (no mutation, just reporting)
# ─────────────────────────────────────────────────────────────────────

def audit(news: pd.DataFrame) -> None:
    """Print a data-quality report on the raw dataset."""
    print(f"\n=== Data Audit ===")
    print(f"Shape: {news.shape[0]:,} rows × {news.shape[1]} columns")

    print("\nMissing values per column:")
    print(news.isna().sum().to_string())

    raw_unique = news["category"].nunique()
    norm_unique = news["category"].str.strip().str.lower().nunique()
    ws_issues = (news["category"] != news["category"].str.strip()).sum()
    print(f"\nCategory labels — unique raw: {raw_unique}, normalized: {norm_unique}")
    print(f"Labels with leading/trailing whitespace: {ws_issues}")

    title_len = news["title"].fillna("").str.split().str.len()
    abst_len = news["abstract"].fillna("").str.split().str.len()
    print(f"\nTitle words    — min {title_len.min()}, "
          f"median {title_len.median():.0f}, max {title_len.max()}")
    print(f"Abstract words — min {abst_len.min()}, "
          f"median {abst_len.median():.0f}, max {abst_len.max()}")
    print(f"Articles with empty abstract: {(abst_len == 0).sum():,} "
          f"({(abst_len == 0).mean():.1%})")

    exact_dupes = news.duplicated(subset=["title", "abstract"]).sum()
    title_dupes = news.duplicated(subset=["title"]).sum()
    print(f"\nExact duplicates (title+abstract): {exact_dupes:,}")
    print(f"Title-only duplicates (likely wire-service republishes): {title_dupes:,}")

    cat_counts = news["category"].value_counts()
    small_cats = cat_counts[cat_counts < 100]
    print(f"\nCategories with <100 articles: {len(small_cats)}")
    if len(small_cats):
        print(small_cats.to_string())


# ─────────────────────────────────────────────────────────────────────
# Stage 3: Cleaning pipeline
# ─────────────────────────────────────────────────────────────────────

def normalize_categories(news: pd.DataFrame, min_size: int = 100) -> pd.DataFrame:
    """Strip whitespace, lowercase, and drop categories below min_size."""
    news = news.copy()
    news["category"] = news["category"].str.strip().str.lower()
    news["subcategory"] = news["subcategory"].fillna("").str.strip().str.lower()

    cat_counts = news["category"].value_counts()
    keep = cat_counts[cat_counts >= min_size].index
    n_before = len(news)
    news = news[news["category"].isin(keep)].reset_index(drop=True)
    dropped_cats = len(cat_counts) - len(keep)
    print(f"[1] Tiny-category drop: removed {n_before - len(news):,} articles "
          f"in {dropped_cats} small categories")
    return news


def build_stopword_set() -> set[str]:
    """Combine NLTK English stopwords with our news-specific additions."""
    nltk.download("stopwords", quiet=True)
    full = set(stopwords.words("english")) | NEWS_STOPWORDS
    print(f"[2] Stopword set: {len(full):,} words "
          f"(English + {len(NEWS_STOPWORDS)} news-specific)")
    return full


def lemmatize_text(news: pd.DataFrame, stopword_set: set[str]) -> pd.DataFrame:
    """Add a 'clean' column with lemmatized, stopword-filtered text."""
    news = news.copy()
    news["title"] = news["title"].fillna("")
    news["abstract"] = news["abstract"].fillna("")
    news["text"] = (news["title"] + " " + news["abstract"]).str.strip()

    print(f"[3] Lemmatizing {len(news):,} articles (spaCy, ~1-3 min on CPU)...")
    nlp = spacy.load("en_core_web_sm",
                     disable=["parser", "ner", "attribute_ruler"])

    def normalize_doc(doc) -> str:
        return " ".join(
            tok.lemma_.lower() for tok in doc
            if tok.is_alpha
            and len(tok.lemma_) > 2
            and tok.lemma_.lower() not in stopword_set
        )

    t0 = time.time()
    news["clean"] = [normalize_doc(d)
                     for d in nlp.pipe(news["text"].tolist(), batch_size=500)]
    print(f"    done in {time.time() - t0:.1f}s")
    return news


def filter_by_length(news: pd.DataFrame,
                     min_tokens: int = 5,
                     max_tokens: int = 500) -> pd.DataFrame:
    """Keep articles whose cleaned text has between min and max tokens."""
    news = news.copy()
    news["text_length"] = news["clean"].str.split().str.len()
    n_before = len(news)
    news = news[news["text_length"].between(min_tokens, max_tokens)]
    print(f"[4] Length filter ({min_tokens}-{max_tokens} tokens): "
          f"dropped {n_before - len(news):,} articles")
    return news


def deduplicate(news: pd.DataFrame) -> pd.DataFrame:
    """Two-stage dedup: exact match on title+abstract, then normalized title."""
    news = news.copy()

    # Exact match
    n_before = len(news)
    news = news.drop_duplicates(subset=["title", "abstract"])
    print(f"[5a] Exact-duplicate drop: removed {n_before - len(news):,} articles")

    # Near-duplicate via normalized title (catches wire-service republishes)
    news["_title_norm"] = (news["title"].str.lower()
                                          .str.replace(r"[^a-z0-9 ]", "", regex=True)
                                          .str.replace(r"\s+", " ", regex=True)
                                          .str.strip())
    n_before = len(news)
    news = news.drop_duplicates(subset=["_title_norm"])
    news = news.drop(columns=["_title_norm"]).reset_index(drop=True)
    print(f"[5b] Near-duplicate drop (normalized title): "
          f"removed {n_before - len(news):,} articles")
    return news


def to_categorical(news: pd.DataFrame) -> pd.DataFrame:
    """Convert category/subcategory to pandas categorical dtype."""
    news = news.copy()
    news["category"] = news["category"].astype("category")
    news["subcategory"] = news["subcategory"].astype("category")
    return news


# ─────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────

def run_pipeline(
    data_dir: str = "MINDlarge_train",
    min_category_size: int = 100,
    min_tokens: int = 5,
    max_tokens: int = 500,
    output_path: str | None = None,
    show_audit: bool = True,
) -> pd.DataFrame:
    """End-to-end: download → load → audit → clean → (optionally) save.

    Returns the cleaned DataFrame.
    """
    download_mind(data_dir)
    news = load_news(data_dir)
    n_start = len(news)

    if show_audit:
        audit(news)

    print(f"\n=== Cleaning Pipeline ===")
    news = normalize_categories(news, min_size=min_category_size)
    stopword_set = build_stopword_set()
    news = lemmatize_text(news, stopword_set)
    news = filter_by_length(news, min_tokens=min_tokens, max_tokens=max_tokens)
    news = deduplicate(news)
    news = to_categorical(news)

    n_end = len(news)
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete: {n_start:,} → {n_end:,} articles "
          f"({(n_start - n_end) / n_start:.1%} removed)")
    print(f"Categories retained: {news['category'].nunique()}")
    print(f"Memory: {news.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"{'=' * 60}")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        news.to_parquet(out)
        print(f"Saved cleaned dataset to {out}")

    return news


def main():
    parser = argparse.ArgumentParser(
        description="Run the MIND news preprocessing pipeline."
    )
    parser.add_argument("--data-dir", default="MINDlarge_train",
                        help="Where MIND data lives (default: MINDlarge_train)")
    parser.add_argument("--output", default="data/news_clean.parquet",
                        help="Output path for cleaned dataset")
    parser.add_argument("--min-category-size", type=int, default=100)
    parser.add_argument("--min-tokens", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--no-audit", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        min_category_size=args.min_category_size,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        output_path=args.output,
        show_audit=not args.no_audit,
    )


if __name__ == "__main__":
    main()
