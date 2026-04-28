# 📰 The Shape of News

A data-mining project on the news articles that don't fit their category. 

> Every news article lives inside a shape: a category, a topic cluster, a neighborhood of similar articles. Most articles sit comfortably in their shape. But about 5% don't fit anywhere cleanly. I examine what they look like, where they sit, and what they have in common.

**👉 Start here: [`main_notebook.ipynb`](./main_notebook.ipynb)**

🎥 **Project video:** https://www.youtube.com/watch?v=6mISUyifGsA

*Anika Garg*

---

## What this project is

When you open MSN News, Google News, or pretty much any news app, every article you see has been sorted into a category by a human editor (Sports, Health, Finance, Lifestyle, News, ...). The MIND dataset I use here has 18 of these categories. Those category tags matter: they're part of how recommenders decide what to show you next, and they're how most filter-bubble research measures whether a feed is diverse.

But if you skim any news feed for a few minutes, you'll find articles that don't quite belong where they're placed. The tag is there, but the article doesn't really look like its neighbors. About 5% of articles in MIND fall into this group. The main question of this project: **what do those articles have in common?**

I tested three standard hypotheses from the literature against each other on real news data:

1. **Miscategorized**: the editor put the article in the wrong bucket.
2. **Crossover**: the article genuinely spans two topics.
3. **Rare but correct**: the article is unusual but in the right category.

---

## Research questions

- **Q0** — Do the editorial categories have distinguishable structure in the text? Which are coherent and which aren't?
- **Q1** — Are the "doesn't fit" articles bridge articles between two categories? (the crossover hypothesis)
- **Q2** — If they're not bridges, where do they actually sit?
- **Q3** — What does an outlier article actually *look like* — concretely, in a case study?

---

## Key findings

- **Editorial categories aren't a uniform taxonomy.** Some have clearly distinct vocabulary (Sports, Finance); others overlap heavily with each other (News, Travel, Lifestyle, Health share 36–44% of their top-50 words). The 15 categories that survive preprocessing don't carve up the text into 15 lexically-distinct regions.
- **Crossover hypothesis: refuted.** Outliers don't sit between categories. UMAP shows them at the *peripheries* of their own category, not in the gaps.
- **Anomalies are longer and entity-richer than typical articles** (~1.6× the length, ~2× as many PERSON mentions, narrower sentiment distribution). This was the opposite of what I expected going in — I was looking for thin wire briefs.
- **The picture this paints is closer to the "miscategorized" hypothesis.** Outliers are well-developed feature pieces in the wrong bucket — like a 53-word piece on the Edgar Allan Poe house in Baltimore that ended up tagged as Finance.
- **Low-purity categories produce more outliers** (Pearson r ≈ -0.42 between K-Means cluster purity and per-category anomaly rate). When a category's content is internally diffuse, off-pattern articles are easier to slip in.

---

## Data

- **Dataset:** [MIND (Microsoft News Dataset)](https://msnews.github.io/) — large training split, 101,527 raw articles, 18 editorial categories.
- **After preprocessing:** 97,313 articles, 15 categories (3 tiny categories with <100 articles were dropped as clustering noise).
- **Source:** auto-downloaded from `https://huggingface.co/datasets/yjw1029/MIND`. No manual data setup required, since the notebook and `src/preprocess.py` handle the download.
- **Citation:** Wu, F., Qiao, Y., Chen, J.-H., Wu, C., Qi, T., Lian, J., Liu, D., Xie, X., Gao, J., Wu, W., & Zhou, M. (2020). *MIND: A Large-scale Dataset for News Recommendation*. ACL.

### Preprocessing pipeline (in order)

1. Strip whitespace and lowercase category labels; drop categories with <100 articles.
2. Build extended stopword set (NLTK English + ~70 news-specific stopwords like *said, told, reuters, getty*).
3. Lemmatize titles + abstracts with spaCy.
4. Filter articles to 5–500 cleaned tokens.
5. Drop exact duplicates on title+abstract; then drop near-duplicates with normalized titles (catches wire-service republishes).
6. Convert categories to pandas categorical dtype.

The full pipeline is implemented inline in the notebook and as a reusable script at [`src/preprocess.py`](./src/preprocess.py).

---

## How to reproduce

This project was built in **Google Colab** with **Python 3.12**.

### Option 1: Run the notebook in Colab (easiest)

1. Open [`main_notebook.ipynb`](./main_notebook.ipynb) in Colab.
2. Run the install cell at the top — it pins `numpy==1.26.4` and `pandas==2.2.2` so the rest of the stack (`gensim`, `node2vec`, `faiss-cpu`, `spacy`) stays compatible.
3. Restart the Colab session once after the first install.
4. Run all cells top-to-bottom. The MIND dataset auto-downloads. Total runtime is ~10–15 minutes.

### Option 2: Run the preprocessing script standalone

```bash
git clone https://github.com/anika-garg/csce676.git
cd csce676
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python src/preprocess.py --output data/news_clean.parquet
```

This downloads MIND, runs the full cleaning pipeline, and saves a parquet file you can load into the notebook.

### Order to read

1. `main_notebook.ipynb`: the full project, top to bottom
2. `checkpoints/checkpoint_1.ipynb`: checkpoint 1, which contains early dataset exploration
3. `checkpoints/checkpoint_2.ipynb`: checkpoint 2, which contains initial project experiments and early findings

---

## Key dependencies

The full pinned environment lives in [`requirements.txt`](./requirements.txt). The main libraries this project depends on:

- `python` 3.12
- `numpy` 1.26.4
- `pandas` 2.2.2
- `scikit-learn` (TF-IDF, K-Means, Isolation Forest, UMAP integration)
- `gensim` (LDA topic modeling)
- `spacy` 3.x with `en_core_web_sm` (lemmatization, named-entity recognition)
- `nltk` (English stopwords)
- `networkx` (similarity graph + community detection)
- `faiss-cpu` (approximate nearest-neighbor search)
- `node2vec` (graph embeddings)
- `umap-learn` (2D projection for visualization)
- `matplotlib`, `seaborn` (plots)
- `textblob` (sentiment polarity)
- `wordcloud` (LDA topic word clouds)

---

## Repo structure

```
news-shape/
├── README.md                  ← you are here
├── main_notebook.ipynb        ← the main deliverable, start here
├── requirements.txt           ← pinned dependencies from Colab
├── checkpoints/
│   ├── checkpoint_1.ipynb     ← early-semester exploration
│   └── checkpoint_2.ipynb     ← midpoint progress
├── src/
│   ├── __init__.py
│   └── preprocess.py          ← reusable preprocessing pipeline
├── data/
│   └── README.md              ← MIND data is auto-downloaded; no manual setup
└── .gitignore
```

---

## Results in one paragraph

I started the project assuming outliers in news data would be **crossover articles** sitting between two categories, and built a graph-based pipeline (FAISS + Node2Vec + community detection) to find them. The data refused to cooperate — outliers turned out to sit at the *peripheries* of their own category, not in the gaps between. So I went looking for what these peripheral articles actually were. My second guess was that they'd be thin wire-service briefs (rare-but-correct articles in vocabulary-distinctive categories). The data refused that one too: anomalies are systematically *longer* and *more* entity-rich than typical articles, with narrower sentiment distributions. The picture that emerged is closer to the **miscategorized hypothesis** — outliers are well-developed pieces tagged with the wrong editorial bucket, like a feature on Baltimore's first female reporter that ended up in Finance. That's a different recommendation problem than "filter the noise"; it suggests these articles need to be *re-routed*, not removed.
