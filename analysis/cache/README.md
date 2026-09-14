# Cached competitor responses

**This directory is tracked on purpose.** Everything else under
`data/processed/` is gitignored and regenerable; this is not. It is the only
asset in the project that costs external API quota to rebuild, and both services
can change or withdraw access at any time.

## What is here

| file | what |
|---|---|
| `headtohead_responses.tar.gz` | 1,394 raw JSON responses — 465 Pl@ntNet, 465 iNaturalist, 464 iNaturalist+geo — one per observation, each a ranked list of `{name, rank, score}`. 5.4 MB uncompressed, 248 KB packed. |
| `headtohead.parquet` | the joined table the analysis scripts actually read. Derived from the JSONs by `plantid/eval/headtohead.py`. |

No credentials, no image data, no personal data — verified before committing.

## Restoring

`plantid/eval/headtohead.py` and the `analysis/` scripts read from
`data/processed/headtohead/` and `data/processed/headtohead.parquet`, which are
gitignored. On a fresh clone:

```bash
mkdir -p data/processed
tar xzf analysis/cache/headtohead_responses.tar.gz -C data/processed
cp analysis/cache/headtohead.parquet data/processed/
```

Re-scoring `COMPETITIVE_FINDINGS.md` then costs **no API quota**:

```bash
PYTHONPATH=. .venv/bin/python -m analysis.compare_h2h
PYTHONPATH=. .venv/bin/python -m analysis.score_bc2_cml4
```

## Why it matters

`COMPETITIVE_FINDINGS.md` is the only measurement anywhere in the four repos that
compares this work against incumbent systems **on identical photographs** —
`bioclip2_cml4` at 0.7720 species top-1 against iNaturalist's 108k-taxa server
model at 0.7871, paired −0.015 [−0.060, +0.030], and ahead at genus 0.974 against
0.914. Without these responses that comparison cannot be rerun, only re-fetched,
and a re-fetch scores a *different* model than the one the finding describes.
