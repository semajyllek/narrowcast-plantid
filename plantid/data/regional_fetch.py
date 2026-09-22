"""Field photographs for a region's species, with the cluster column carried.

Takes a survey from `plantid.data.regions` and fetches images per species into a
manifest `narrowcast` can consume. Modelled on `analysis/herbarium_fetch.py`,
which is the only fetcher here that is resumable and argparse'd; the `analysis/`
iNaturalist fetchers keep region, species list and output path as module globals
and have no CLI.

Three things this does that the older fetchers do not.

**The cluster column is carried.** An occurrence bundles several photographs of
one individual plant, and those are not independent observations. Every split and
every bootstrap in this project keys on a cluster column, and
`analysis/export_for_narrowcast.py` deliberately omits one because Pl@ntNet has no
per-individual grouping. Regional data *does* -- the GBIF occurrence key, or the
iNaturalist observation id -- so it is recorded here and must survive into the
`.npz`. Without it every regional bundle reports anticonservative intervals, and
the card will say so.

**Live plants, not herbarium sheets.** `basisOfRecord=HUMAN_OBSERVATION` by
default. A pressed specimen scan is a legitimate occurrence and looks nothing like
a photograph taken on a phone, which is the acquisition shift `DOMAIN_SHIFT_FINDINGS`
warns costs a 17.9 MB encoder far more than a large one.

**Licences are recorded per image.** A product redistributes what it ships. GBIF
and iNaturalist both carry per-record licences, several of which are
non-commercial, and that is a decision for whoever ships rather than something to
discover later.

Usage:
    PYTHONPATH=. .venv/bin/python -m plantid.data.regional_fetch \
        --survey data/processed/regions/oregon_gbif.json \
        --out data/processed/regions/oregon --per-species 30 --max-species 50
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

from plantid.data.curation import curated_name

GBIF_OCC = "https://api.gbif.org/v1/occurrence/search"
HEADERS = {"User-Agent": "plantid-research/0.1 (species identification evaluation)"}
MAX_PX = 512
SLEEP = 0.2
RETRIES = 4
PAGE = 300          # GBIF's per-request cap
MAX_SCAN = 3000     # stop scanning a species rather than page forever
BACKOFF = 3.0

# Licences that permit redistribution in a commercial product. Anything else is
# fetched only with `--any-licence`, and is recorded either way so the decision is
# visible rather than discovered at ship time.
#
# The NC question is the live one: on a sample of Oregon Trillium occurrences, 16
# of 20 were CC BY-NC. A non-commercial licence does not stop research use and
# does stop shipping, so which bucket a region's images fall into is a product
# fact worth knowing before fetching tens of thousands of them.
REDISTRIBUTABLE = ("CC0", "CC_BY")


def parse_licence(url: str | None) -> str:
    """GBIF returns a licence URL; turn it into a comparable token.

    The URL ends `/legalcode`, so the naive last-path-segment read returns
    "LEGALCODE" for every record and silently classifies nothing -- which is what
    happened first and is why this is a named function with a test.
    """
    if not url:
        return "UNKNOWN"
    parts = [p for p in url.lower().split("/") if p and p != "legalcode"]
    if "publicdomain" in parts and "zero" in parts:
        return "CC0"
    if "licenses" in parts:
        i = parts.index("licenses")
        if i + 1 < len(parts):
            return "CC_" + parts[i + 1].upper().replace("-", "_")
    return "UNKNOWN"


def _get(url, params, retries=RETRIES):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                last = r.status_code
                time.sleep(BACKOFF * (2 ** attempt))
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            last = e
            time.sleep(BACKOFF * (2 ** attempt))
    raise RuntimeError(f"GBIF did not answer after {retries} attempts: {last}")


def occurrences(species: str, want: int, state_province=None, country="US",
                basis="HUMAN_OBSERVATION", open_only=True) -> list[dict]:
    """Image-bearing occurrences of one species in one region.

    One entry per *photograph*, but `cluster` is the occurrence key, so several
    photographs of the same plant share a cluster and cannot be split across a
    train/test boundary.

    The record's own resolved name is checked against the query: GBIF's
    `scientificName` search admits synonyms, and a record filed under a name the
    catalogue does not use is a label mismatch, not a hard example.
    """
    # **Paged, and that is not an optimisation.** This used to take a single page
    # of `want * 3` records and keep whatever survived the synonym and licence
    # filters. GBIF returns a page in no useful order, and for a common species
    # the first sixty records are often one dataset under one licence -- so when
    # that licence was non-redistributable, almost nothing survived.
    #
    # The effect was backwards and invisible: *Polystichum munitum* has 7,652
    # Oregon observations and yielded ONE usable plant, *Gaultheria shallon*
    # 6,511 and yielded two, while obscure species got ten. Across the region the
    # correlation between how often a plant is photographed and how many distinct
    # plants were fetched was **-0.171** -- the bank was thinnest exactly where a
    # user is most likely to point their phone, and 75 species were dropped from
    # the model entirely for thinness.
    params = {"scientificName": species, "mediaType": "StillImage",
              "hasCoordinate": "true", "country": country,
              "basisOfRecord": basis, "limit": PAGE}
    if state_province:
        params["stateProvince"] = state_province

    results, offset = [], 0
    while offset < MAX_SCAN:
        page = _get(GBIF_OCC, {**params, "offset": offset})
        recs = page.get("results", [])
        results += recs
        # enough *distinct, usable* occurrences is the stopping condition, not
        # enough records scanned
        usable = {str(r.get("key")) for r in results
                  if r.get("species") == species
                  and (not open_only
                       or parse_licence(r.get("license")) in REDISTRIBUTABLE)
                  and (r.get("media") or [])}
        if len(usable) >= want or page.get("endOfRecords") or not recs:
            break
        offset += PAGE

    out = []
    for rec in results:
        if rec.get("species") != species:
            continue
        licence = parse_licence(rec.get("license"))
        if open_only and licence not in REDISTRIBUTABLE:
            continue
        key = str(rec.get("key"))
        for i, m in enumerate(rec.get("media") or []):
            url = m.get("identifier")
            if not url or not (m.get("format") or "").startswith("image/"):
                continue
            out.append({
                "species_name": species,
                "cluster": key,                       # the occurrence: one plant
                "photo_index": i,
                "url": url,
                "licence": licence,
                "institution": rec.get("institutionCode") or "?",
                "recorded_by": rec.get("recordedBy") or "?",
                "gbif_key": key,
            })
        if len({r["cluster"] for r in out}) >= want:
            break
    return out


def grab(url: str, dest: Path) -> bool:
    from PIL import Image
    tmp = dest.with_suffix(".tmp")
    try:
        r = requests.get(url, headers=HEADERS, timeout=90)
        r.raise_for_status()
        tmp.write_bytes(r.content)
        im = Image.open(tmp).convert("RGB")
        im.thumbnail((MAX_PX, MAX_PX))
        im.save(dest, "JPEG", quality=90)
        return True
    except Exception:
        return False
    finally:
        tmp.unlink(missing_ok=True)


def load_species(survey_path: Path, min_obs: int, max_species: int | None) -> list[str]:
    s = json.loads(Path(survey_path).read_text())
    sp = [x for x in s["species"] if x["n_obs"] >= min_obs]
    names = [x["curated"] for x in sp]
    # Shuffled rather than alphabetical: alphabetical is ordered by genus, so a
    # partial run would be a biased sample of one part of the flora rather than a
    # random slice of the region.
    import random
    random.Random(0).shuffle(names)
    return names[:max_species] if max_species else names


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey", required=True, help="JSON from plantid.data.regions")
    ap.add_argument("--out", required=True, help="image directory for this region")
    ap.add_argument("--per-species", type=int, default=30,
                    help="occurrences (not photographs) per species")
    ap.add_argument("--min-obs", type=int, default=100,
                    help="skip species with fewer regional records than this")
    ap.add_argument("--max-species", type=int)
    ap.add_argument("--state", default=None, help="GBIF stateProvince to restrict to")
    ap.add_argument("--country", default="US")
    ap.add_argument("--any-licence", action="store_true",
                    help="fetch regardless of licence. Off by default: a shipped "
                         "product redistributes these images.")
    ap.add_argument("--sleep", type=float, default=SLEEP)
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.parquet"

    done: set[str] = set()
    rows: list[dict] = []
    if manifest_path.exists():
        prev = pd.read_parquet(manifest_path)
        rows = prev.to_dict("records")
        done = set(prev["species_name"])
        print(f"resuming: {len(done)} species already fetched", flush=True)

    species = load_species(a.survey, a.min_obs, a.max_species)
    todo = [s for s in species if s not in done]
    print(f"{len(species)} species requested, {len(todo)} to fetch "
          f"({'open licences only' if not a.any_licence else 'any licence'})",
          flush=True)

    for n, sp in enumerate(todo, 1):
        try:
            recs = occurrences(sp, a.per_species, state_province=a.state,
                               country=a.country, open_only=not a.any_licence)
        except RuntimeError as e:
            print(f"  [{n}/{len(todo)}] {sp}: {e}", flush=True)
            continue
        sp_dir = out_dir / sp.replace(" ", "_")
        sp_dir.mkdir(exist_ok=True)
        kept = 0
        for r in recs:
            dest = sp_dir / f"{r['cluster']}_{r['photo_index']}.jpg"
            if dest.exists() or grab(r["url"], dest):
                # Relative to the region directory, so that directory is
                # self-contained and portable: move it anywhere and the manifest
                # still resolves. It was relative to out_dir.parent.parent, which
                # is unguessable from the manifest alone and duly produced a
                # `regions/regions/...` on first use.
                rows.append({**r, "local_path": str(dest.relative_to(out_dir))})
                kept += 1
            time.sleep(a.sleep)
        n_clusters = len({r["cluster"] for r in recs})
        print(f"  [{n}/{len(todo)}] {sp}: {kept} photos over {n_clusters} plants",
              flush=True)
        # Rewritten every species so an interrupted run keeps everything so far.
        pd.DataFrame(rows).to_parquet(manifest_path, index=False)

    if rows:
        df = pd.DataFrame(rows)
        print(f"\n{manifest_path}: {len(df)} photos, "
              f"{df['species_name'].nunique()} species, "
              f"{df['cluster'].nunique()} distinct plants")
        print(f"photos per plant: median "
              f"{df.groupby('cluster').size().median():.1f}")
        print("licences: " + ", ".join(f"{k} {v}" for k, v in
                                       df['licence'].value_counts().items()))


if __name__ == "__main__":
    main()
