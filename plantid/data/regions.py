"""What plants are in a place, and which of them have enough data to learn.

This is the piece a regional product is built on and the piece that did not exist.
`analysis/common_oregon_fetch.py` is a hand-typed list of twenty species; the
region itself is the integer literal `"place_id": 10` inside
`analysis/safety_fetch.py`, copied into two more files. The note in `CLAUDE.md`
that Oregon has "4,570 research-grade species, 1,175 with >=100 observations"
records an ad-hoc query that was never committed. Nothing here called
`species_counts` before.

The endpoint returns, for a place, every taxon observed in it with a count. That
count is the data-sufficiency signal a regional bundle needs: `PHASE0_FINDINGS.md`
measures the adapted encoder as saturated at **two** training images per species
and stock at 16-32, so "does this species have enough images" is a threshold on
this number, not a guess.

Two deliberate choices:

**Species rank only.** `rank=species` excludes genus- and family-level
observations, which are a large share of casual plant records and cannot train a
species head.

**Counts are observations, not photographs.** An observation bundles several
photos of one individual, which is also the cluster unit -- so `n_obs` is closer
to "how many independent examples" than a photo count would be, and that is the
quantity the splits and bootstraps key on.

Usage:
    python -m plantid.data.regions --place 10 --out data/processed/regions/oregon.json
    python -m plantid.data.regions --place 10 --min-obs 100 --print-only
"""

import argparse
import json
import time
from pathlib import Path

import requests

from plantid.data.curation import curated_name

OBS_API = "https://api.inaturalist.org/v1/observations/species_counts"
PLACES_API = "https://api.inaturalist.org/v1/places"
HEADERS = {"User-Agent": "plantid-research/0.1 (species identification evaluation)"}
SLEEP = 1.1          # iNat asks for <= 60 requests/minute
PER_PAGE = 500       # the endpoint's maximum
PLANTAE = 47126      # iNat taxon id for the kingdom

# Named so a region is declared data rather than an integer literal in three
# files. iNat place ids are stable; `--place` also accepts a raw id for anywhere
# not listed here.
PLACES = {
    "oregon": 10,
    "washington": 46,
    "california": 14,
    "british-columbia": 7085,
    "united-kingdom": 6857,
}


RETRIES = 5
BACKOFF = 4.0


def _get(url, params, sleep=SLEEP, retries=RETRIES):
    """GET with backoff on the failures a long survey actually hits.

    A full survey is dozens of sequential requests over several minutes, so a
    single transient 503 should not lose the run -- iNaturalist returns 503
    "downtime" for maintenance windows, and 429 when a client outruns the rate
    limit. Both are retried; a 404 or a malformed query is not, because retrying
    those just repeats the same mistake more slowly.

    This exists because an HF Hub disconnect killed a twelve-hour embedding run
    in this project, and the cost of that was entirely in not having retried.
    """
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                last = requests.exceptions.HTTPError(
                    f"{r.status_code} from {url}", response=r)
                wait = BACKOFF * (2 ** attempt)
                print(f"    {r.status_code} from iNat; retrying in {wait:.0f}s "
                      f"({attempt + 1}/{retries})", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            time.sleep(sleep)
            return r.json()
        except requests.exceptions.RequestException as e:
            last = e
            wait = BACKOFF * (2 ** attempt)
            print(f"    {type(e).__name__}; retrying in {wait:.0f}s "
                  f"({attempt + 1}/{retries})", flush=True)
            time.sleep(wait)
    raise SystemExit(
        f"iNaturalist did not answer after {retries} attempts: {last}\n"
        "If every endpoint returns 503 the API is in a maintenance window; "
        "the survey is resumable, so rerun it later.")


def place_name(place_id: int) -> str | None:
    """Human-readable name for a place id, for recording in the output."""
    try:
        res = _get(f"{PLACES_API}/{place_id}", {})["results"]
        return res[0]["display_name"] if res else None
    except Exception:
        return None


def species_counts(place_id: int, taxon_id: int = PLANTAE,
                   quality_grade: str = "research", max_pages: int = 60,
                   verbose: bool = True) -> list[dict]:
    """Every species observed in a place, with its observation count.

    Descending by count, so a truncated run keeps the well-observed species
    rather than a random slice -- which matters because `max_pages` is a guard
    against a place with tens of thousands of taxa, not an expected stopping
    point.
    """
    out, page = [], 1
    while page <= max_pages:
        data = _get(OBS_API, {
            "place_id": place_id, "taxon_id": taxon_id,
            "quality_grade": quality_grade, "rank": "species",
            "per_page": PER_PAGE, "page": page,
        })
        results = data.get("results", [])
        if not results:
            break
        for r in results:
            t = r.get("taxon") or {}
            if t.get("rank") != "species" or not t.get("name"):
                continue
            out.append({
                "taxon_id": t["id"],
                "name": t["name"],
                "curated": curated_name(t["name"]) or t["name"],
                "common_name": t.get("preferred_common_name"),
                "n_obs": int(r.get("count", 0)),
            })
        total = data.get("total_results", 0)
        if verbose:
            print(f"  page {page}: {len(out)} species so far (of {total})", flush=True)
        if len(out) >= total or len(results) < PER_PAGE:
            break
        page += 1
    return out


def usable(species: list[dict], min_obs: int = 20) -> list[dict]:
    """Species with enough independent observations to train a head on.

    `min_obs` is an observation threshold, and observations are the cluster unit,
    so this is a count of independent examples rather than photographs. The right
    value depends on the encoder: PHASE0_FINDINGS puts the adapted tower at
    saturation by ~2 training images per species and stock S2 at 16-32, and a
    train/test split needs both sides, so 20 is a conservative default rather
    than a measured floor.
    """
    return [s for s in species if s["n_obs"] >= min_obs]


def survey(place_id: int, min_obs: int = 20, taxon_id: int = PLANTAE,
           verbose: bool = True) -> dict:
    """The whole pool for a place, plus the counts that describe it."""
    allsp = species_counts(place_id, taxon_id=taxon_id, verbose=verbose)
    keep = usable(allsp, min_obs)
    by_genus: dict[str, int] = {}
    for s in keep:
        by_genus[s["curated"].split()[0]] = by_genus.get(s["curated"].split()[0], 0) + 1
    return {
        "place_id": place_id,
        "place_name": place_name(place_id),
        "taxon_id": taxon_id,
        "min_obs": min_obs,
        "n_species_total": len(allsp),
        "n_species_usable": len(keep),
        "n_genera_usable": len(by_genus),
        # The crowding a region *forces*: genera with several usable members are
        # where a user-chosen list will accidentally pick siblings, which costs
        # far more label share than any encoder choice does.
        "crowded_genera": dict(sorted(((g, n) for g, n in by_genus.items() if n >= 4),
                                      key=lambda kv: -kv[1])),
        "species": keep,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", required=True,
                    help=f"iNat place id, or one of: {', '.join(sorted(PLACES))}")
    ap.add_argument("--min-obs", type=int, default=20,
                    help="observations required to call a species usable (default 20)")
    ap.add_argument("--out", help="write the survey as JSON")
    ap.add_argument("--print-only", action="store_true")
    a = ap.parse_args()

    place_id = PLACES.get(a.place.lower(), None)
    if place_id is None:
        if not a.place.isdigit():
            raise SystemExit(f"unknown place {a.place!r}; give an id or one of "
                             f"{', '.join(sorted(PLACES))}")
        place_id = int(a.place)

    print(f"surveying place {place_id} (plants, research grade, species rank)", flush=True)
    s = survey(place_id, min_obs=a.min_obs)
    print(f"\n{s['place_name'] or place_id}: {s['n_species_total']} species observed, "
          f"{s['n_species_usable']} with >= {a.min_obs} observations, "
          f"across {s['n_genera_usable']} genera")
    if s["crowded_genera"]:
        top = list(s["crowded_genera"].items())[:6]
        print("  genera with 4+ usable species (where a user list will pick siblings): "
              + ", ".join(f"{g} ({n})" for g, n in top))
    print("\n  top by observation count:")
    for sp in s["species"][:10]:
        print(f"    {sp['curated']:<38} {sp['n_obs']:>7}  {sp['common_name'] or ''}")

    if a.out and not a.print_only:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(s, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
