# data/

Git ignores everything in this folder except this file. Do not commit posts, samples or scores. Posts
contain personal data, and platform terms restrict the redistribution of collected posts. The offline
demo and the tests use the synthetic generator (`twitter-to-x synth`) and need no download.

## Supported sources

| KIND | Source | Where to get it | Expected file and columns | Notes |
|---|---|---|---|---|
| `sentiment140` | Sentiment140 (2009) | Kaggle `kazanova/sentiment140` | `training.1600000.processed.noemoticon.csv`, latin-1, no header: `target, id, date, flag, user, text` | The file is SORTED by label. Rows 1 to 800 000 are negative. Never slice it: `prepare` samples at random |
| `x_parquet` | A public X dataset chunk | Hugging Face, for example `arrmlet/x_dataset_205` | parquet with `text`, `datetime` and `username_encoded` (needs the `parquet` extra) | Check the dataset card and its terms |
| `csv` | Your own collection (for example from the X API) | Collect under the platform terms | `text`, `created_at` and optional `author`, `post_id` | Use the same query design for both periods |

The `pre` and `post` periods come from `created_at` and the cutoff (`TTX_CUTOFF`, default `2022-10-27`).

**A sound design.** Sentiment140 is from 2009 and its labels come from emoticons. A comparison of 2009
with 2024 mixes the platform change with 15 years of language change. Collect both periods with the same
keywords or topics and similar sizes, for example 2021-2022 against 2023-2024.

## Privacy

- The loaders hash the author column with HMAC-SHA256 and the secret `TTX_SALT`. Without a salt, each
  run uses a new random salt, so hashes cannot be linked across runs.
- Handles in the text become `@user` and links become `http`.
- `analyze` writes `scores.csv` without the text column.

## Files that `prepare` writes

| File | Contents |
|---|---|
| `data/prepared/posts.csv` | `post_id, text, created_at, source, author_hash, period` and optional `true_*` columns |
| `data/prepared/manifest.json` | cutoff, seed, rows and sources per period, date ranges, filter counts per rule and period |
