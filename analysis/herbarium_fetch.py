"""Herbarium specimen images for catalogue species, from the GBIF occurrence API.

Pre-registered in `HERBARIUM_PREREG.md`. The Tier 1 acquisition-shift probe:
institutionally determined (a botanist's identification on the sheet, not a crowd
vote) and selectively fetchable by species, which is what makes it affordable
where the FGVC9 bulk download is not.

Images are resized on arrival and the original discarded -- a herbarium sheet
scan can be tens of megabytes and nothing here needs that.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.herbarium_fetch --per-species 8
"""

import argparse
import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

import certifi
import pandas as pd

from plantid.config import DATA_PROCESSED
from plantid.data.curation import curated_name

API = "https://api.gbif.org/v1/occurrence/search"
CTX = ssl.create_default_context(cafile=certifi.where())
OUT = DATA_PROCESSED / "images_herbarium"
MAX_PX = 512
UA = {"User-Agent": "narrowcast-plantid/1.0 (research; contact via github.com/semajyllek)"}


def catalogue_species(cache_dir=DATA_PROCESSED):
    c = pd.read_parquet(cache_dir / "catalog_index.parquet")
    return sorted({n for n in (curated_name(x) for x in c["species_name"].unique()) if n})


def records(species: str, want: int):
    """Occurrence records with a still image, oldest API page first.

    `species` is checked against the record's own resolved name: GBIF's
    scientificName search admits synonyms, and a sheet filed under a name the
    catalogue does not use is a label mismatch rather than a hard example.
    """
    q = urllib.parse.urlencode({"scientificName": species,
                                "basisOfRecord": "PRESERVED_SPECIMEN",
                                "mediaType": "StillImage",
                                "limit": max(want * 3, 30)})
    req = urllib.request.Request(f"{API}?{q}", headers=UA)
    with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
        results = json.load(r).get("results", [])
    out = []
    for rec in results:
        if rec.get("species") != species:
            continue
        for m in rec.get("media") or []:
            url = m.get("identifier")
            if url and (m.get("format") or "").startswith("image/"):
                out.append({"url": url, "institution": rec.get("institutionCode") or "?",
                            "country": rec.get("countryCode") or "?",
                            "key": str(rec.get("key"))})
                break
        if len(out) >= want:
            break
    return out


def grab(url: str, dest: Path) -> bool:
    from PIL import Image
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=90, context=CTX) as r:
            dest.with_suffix(".tmp").write_bytes(r.read())
        im = Image.open(dest.with_suffix(".tmp")).convert("RGB")
        im.thumbnail((MAX_PX, MAX_PX))
        im.save(dest, "JPEG", quality=90)
        return True
    except Exception:
        return False
    finally:
        dest.with_suffix(".tmp").unlink(missing_ok=True)


def main(per_species: int, max_species: int | None, sleep: float):
    OUT.mkdir(parents=True, exist_ok=True)
    species = catalogue_species()
    if max_species:
        species = species[:max_species]
    manifest_path = DATA_PROCESSED / "herbarium_index.parquet"
    done = set()
    if manifest_path.exists():
        prev = pd.read_parquet(manifest_path)
        done = set(prev["species_name"])
        rows = prev.to_dict("records")
    else:
        rows = []

    t0 = time.time()
    for i, sp in enumerate(species):
        if sp in done:
            continue
        d = OUT / sp.replace(" ", "_")
        d.mkdir(exist_ok=True)
        try:
            recs = records(sp, per_species)
        except Exception as e:
            print(f"  {sp}: query failed ({type(e).__name__})", flush=True)
            continue
        got = 0
        for rec in recs:
            dest = d / f"{rec['key']}.jpg"
            if dest.exists() or grab(rec["url"], dest):
                rows.append({"species_name": sp, "local_path": str(dest.relative_to(DATA_PROCESSED)),
                             "institution": rec["institution"], "country": rec["country"],
                             "gbif_key": rec["key"]})
                got += 1
            time.sleep(sleep)
        print(f"[{i + 1}/{len(species)}] {sp}: {got}/{len(recs)}  "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(manifest_path, index=False)

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} images over {df['species_name'].nunique()} species -> {manifest_path}")
    print(df["institution"].value_counts().head(8).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-species", type=int, default=8)
    ap.add_argument("--max-species", type=int)
    ap.add_argument("--sleep", type=float, default=0.15)
    a = ap.parse_args()
    main(a.per_species, a.max_species, a.sleep)
