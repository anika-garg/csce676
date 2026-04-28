# Data

This folder is intentionally empty in the repo — the [MIND dataset](https://msnews.github.io/) is too large to commit and gets downloaded automatically when you run the project.

## Source

**MIND: MIcrosoft News Dataset** — large training split.

- **URL:** [https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_train.zip](https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_train.zip)
- **Size:** ~530 MB unzipped
- **Articles:** 101,527 raw → 97,313 after preprocessing
- **Categories:** 18 raw → 15 after dropping tiny categories with <100 articles

## How to populate this folder

The dataset downloads itself the first time you run either the main notebook or the preprocessing script. No manual steps required.

**From the main notebook:**

The download cell near the top of `main_notebook.ipynb` handles it. The MIND files extract to `MINDlarge_train/` at the repo root.

**From the standalone script:**

```bash
python src/preprocess.py --data-dir data/MINDlarge_train --output data/news_clean.parquet
```

The `--data-dir` flag tells the script to extract MIND directly into this folder instead of the repo root.

## What ends up here after running

After the pipeline runs, this folder will contain:

- `MINDlarge_train/` — the extracted MIND files
  - `news.tsv` — article metadata (ID, category, subcategory, title, abstract, URL, entities)
  - `behaviors.tsv` — user click behavior (not used in this project)
  - `entity_embedding.vec` — pre-trained Wikidata entity embeddings (not used)
  - `relation_embedding.vec` — pre-trained relation embeddings (not used)
- `news_clean.parquet` — the cleaned dataset used by the main notebook (~50 MB)

All of these are gitignored.

## Schema (news.tsv)

The raw MIND `news.tsv` file is tab-separated with no header. Columns (assigned in code):

| # | Column | Description |
|---|---|---|
| 1 | `id` | Unique article ID (e.g., `N55528`) |
| 2 | `category` | Editorial category (e.g., `news`, `sports`) |
| 3 | `subcategory` | Editorial subcategory (e.g., `newsworld`, `football_nfl`) |
| 4 | `title` | Article headline |
| 5 | `abstract` | Article abstract / lede paragraph |
| 6 | `url` | Source URL |
| 7 | `entities_title` | JSON-encoded Wikidata entities mentioned in title |
| 8 | `entities_abstract` | JSON-encoded Wikidata entities mentioned in abstract |

This project uses columns 2–5; the entity columns are kept for reference but not used in the analysis.

## Citation

Wu, F., Qiao, Y., Chen, J.-H., Wu, C., Qi, T., Lian, J., Liu, D., Xie, X., Gao, J., Wu, W., & Zhou, M. (2020). *MIND: A Large-scale Dataset for News Recommendation*. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL).
