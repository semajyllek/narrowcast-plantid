"""Unseen-species probe set: iNaturalist observations of plants outside Pl@ntNet-300K.

Pre-registered in the addendum to `ADAPT_PREREG.md`. The adapted tower has never
seen a species outside Pl@ntNet's 1,081, so these are genuinely unseen -- and
they come from iNaturalist, a source the adaptation never touched, which makes
this a harder test than the notebook's same-source probe.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.probe_fetch
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import safety_fetch as sf

sf.SPECIES = [s for s in Path("/tmp/probe_species.txt").read_text().split("\n") if s]
sf.OUT = Path(__file__).parent / "probe_unseen"
sf.MAX_OBS = 25
sf.MAX_PHOTOS = 2


def _global_observations(name):
    """Same as safety_fetch.observations but worldwide -- these species are not
    Oregon plants and place_id=10 would return almost nothing."""
    import time

    import requests

    rows, page = [], 1
    while len(rows) < sf.MAX_OBS and page <= 3:
        r = requests.get(sf.OBS, params={
            "taxon_name": name, "quality_grade": "research", "photos": "true",
            "rank": "species", "per_page": 100, "page": page, "order_by": "votes",
        }, headers=sf.H, timeout=60)
        res = r.json().get("results", [])
        if not res:
            break
        for o in res:
            if ((o.get("taxon") or {}).get("name")) != name:
                continue
            urls = [p["url"].replace("/square", "/medium")
                    for p in (o.get("photos") or [])[:sf.MAX_PHOTOS] if p.get("url")]
            if urls:
                rows.append({"obs_id": o["id"], "species_name": name, "urls": urls})
        page += 1
        time.sleep(sf.SLEEP)
    return rows[:sf.MAX_OBS]


sf.observations = _global_observations

if __name__ == "__main__":
    sf.main()
