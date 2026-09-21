"""A field guide's plant list -> a species list the fetch pipeline can take.

The premise of this whole project is a user who *already knows* which plants they
care about. A regional field guide is the best expression of that: someone who
knows the place chose the list, which beats "the 200 most-photographed species"
on relevance and on the thing that actually decides the product — whether the
plant in front of you is on the list at all.

What a book gives you is not what the pipeline wants. Books lead with common
names ("salal", "sword fern"), sometimes carry a binomial a taxonomic generation
old, and sometimes name a plant at a granularity no classifier can honour
("buttercup" is 25 species in Oregon alone).

## What this does, and what it refuses to do

Every input name lands in exactly one of four buckets, and only the first is used:

  resolved     one species, confidently
  ambiguous    several plausible species — REPORTED, never guessed between
  not_in_region  a real species that does not occur where you are surveying
  not_found    nothing matched

**Ambiguity is reported rather than resolved.** Silently picking the most
photographed *Ranunculus* for "buttercup" would produce a catalogue whose labels
do not mean what the book meant, and the user would never see it happen. The
report names the candidates so a person can choose.

## Local first

The regional survey (`plantid.data.regions`) already carries every species in the
area with its common name and observation count. Matching against that first
means most names resolve with no network call at all, and — more importantly —
**anything that resolves is guaranteed to be a plant that actually occurs there**.
iNaturalist is consulted only for names the survey cannot place, and its answer
is still checked against the survey before being accepted.

Usage:
    python -m plantid.data.booklist --names my_book_list.txt \\
        --survey data/processed/regions/oregon_gbif.json \\
        --out data/processed/regions/my_list.json
"""

import argparse
import json
import re
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

INAT_TAXA = "https://api.inaturalist.org/v1/taxa"
PLANTAE = 47126
BINOMIAL = re.compile(r"^[A-Z][a-z]+ [a-z][a-z-]+$")
# Below this share of the candidates' observations, the top match is not a
# winner and the name is ambiguous. A book that says "Oregon grape" in the
# Willamette Valley means one plant; a book that says "buttercup" does not.
DOMINANCE = 0.70


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def read_names(path: Path) -> list[str]:
    """One name per line. `#` comments and blank lines ignored; a trailing
    `(anything)` is dropped, since field guides annotate entries."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        line = re.sub(r"\s*\([^)]*\)\s*$", "", line).strip()
        if line:
            out.append(line)
    return out


def load_survey(path: Path) -> list[dict]:
    d = json.loads(Path(path).read_text())
    return d["species"] if isinstance(d, dict) else d


def _norm(s: str) -> str:
    return re.sub(r"[^a-z ]", " ", str(s).lower()).split() and " ".join(
        re.sub(r"[^a-z ]", " ", str(s).lower()).split()) or ""


def match_local(name: str, survey: list[dict]) -> list[dict]:
    """Candidates from the regional survey alone. Exact matches win outright."""
    n = _norm(name)
    sci = [s for s in survey if _norm(s.get("curated") or s["name"]) == n
           or _norm(s["name"]) == n]
    if sci:
        return sci
    exact = [s for s in survey if _norm(s.get("common_name")) == n]
    if exact:
        return exact
    # a book's "western sword fern" against a survey's "sword fern", or the
    # reverse: accept containment in either direction, then let DOMINANCE decide
    return [s for s in survey
            if (c := _norm(s.get("common_name"))) and (n in c or c in n)]


def inat_any(name: str, ctx) -> list[str]:
    """Whatever iNaturalist thinks the name is, region ignored."""
    q = urllib.parse.urlencode({"q": name, "rank": "species",
                                "taxon_id": PLANTAE, "per_page": 5})
    try:
        with urllib.request.urlopen(f"{INAT_TAXA}?{q}", timeout=30, context=ctx) as r:
            return [t["name"] for t in json.load(r)["results"]]
    except Exception:
        return []


def match_inat(name: str, survey: list[dict], ctx) -> list[dict]:
    """Ask iNaturalist, then keep only answers the survey also has.

    The second half is the important half: iNat's search is global and fuzzy, so
    "sword fern" will happily return a Florida species. A name is only resolved
    if the plant occurs where you are actually surveying.
    """
    q = urllib.parse.urlencode({"q": name, "rank": "species",
                                "taxon_id": PLANTAE, "per_page": 10})
    try:
        with urllib.request.urlopen(f"{INAT_TAXA}?{q}", timeout=30, context=ctx) as r:
            results = json.load(r)["results"]
    except Exception:
        return []
    by_name = {_norm(s.get("curated") or s["name"]): s for s in survey}
    return [by_name[_norm(t["name"])] for t in results if _norm(t["name"]) in by_name]


def resolve(names, survey, use_inat=True):
    ctx = _ctx() if use_inat else None
    out = {"resolved": {}, "ambiguous": {}, "not_in_region": [], "not_found": []}
    survey_sci = {_norm(s.get("curated") or s["name"]) for s in survey}
    for name in names:
        cands = match_local(name, survey)
        if not cands and use_inat:
            cands = match_inat(name, survey, ctx)
        if not cands:
            # "a real plant, just not here" is a different thing to tell a user
            # than "no idea what that is", and only the first is their mistake to
            # fix by changing region rather than by changing the name.
            elsewhere = (BINOMIAL.match(name) and _norm(name) not in survey_sci)
            if not elsewhere and use_inat:
                known = inat_any(name, ctx)
                if known:
                    out["not_in_region"].append(f"{name} (iNat: {known[0]})")
                    continue
            (out["not_in_region"] if elsewhere else out["not_found"]).append(name)
            continue
        cands = sorted(cands, key=lambda s: -s.get("n_obs", 0))
        total = sum(c.get("n_obs", 0) for c in cands) or 1
        top = cands[0]
        if len(cands) == 1 or top.get("n_obs", 0) / total >= DOMINANCE:
            out["resolved"][name] = top.get("curated") or top["name"]
        else:
            out["ambiguous"][name] = [
                {"species": c.get("curated") or c["name"],
                 "common_name": c.get("common_name"), "n_obs": c.get("n_obs", 0)}
                for c in cands[:6]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True, help="one name per line; # comments ok")
    ap.add_argument("--survey", required=True, help="JSON from plantid.data.regions")
    ap.add_argument("--out", required=True, help="where to write the resolved list")
    ap.add_argument("--no-inat", action="store_true",
                    help="survey only, no network")
    a = ap.parse_args()

    names = read_names(Path(a.names))
    survey = load_survey(Path(a.survey))
    r = resolve(names, survey, use_inat=not a.no_inat)

    species = sorted(set(r["resolved"].values()))
    keep = {s["curated"] or s["name"]: s for s in survey}
    payload = {"source": str(a.names), "survey": str(a.survey),
               "n_input": len(names), "species": [
                   {k: v for k, v in keep[s].items()} for s in species if s in keep],
               "resolved": r["resolved"], "ambiguous": r["ambiguous"],
               "not_in_region": r["not_in_region"], "not_found": r["not_found"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=1))

    print(f"{len(names)} names -> {len(species)} species  ({a.out})")
    if r["ambiguous"]:
        print(f"\n{len(r['ambiguous'])} AMBIGUOUS — pick one and put the binomial "
              "in your list, or drop the entry:")
        for n, c in r["ambiguous"].items():
            print(f"  {n!r}")
            for x in c:
                print(f"      {x['species']:32s} {x['common_name'] or '-':28s} "
                      f"{x['n_obs']:>7,} obs")
    for key, label in (("not_in_region", "NOT IN THIS REGION"),
                       ("not_found", "NOT FOUND")):
        if r[key]:
            print(f"\n{len(r[key])} {label}: " + ", ".join(repr(x) for x in r[key]))


if __name__ == "__main__":
    main()
