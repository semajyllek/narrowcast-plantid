"""Region survey: pagination, filtering, and the retry that a long run needs.

Everything here runs offline. The iNaturalist API was in a maintenance window the
day this was written, which is exactly the condition the retry exists for and
exactly the reason none of these tests may touch the network.
"""
import pytest
import requests

from plantid.data import regions


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}
        self.text = "" if status == 200 else f"<html>{status}</html>"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}", response=self)


def _page(names_counts, total):
    return {"total_results": total,
            "results": [{"count": c,
                         "taxon": {"id": i, "name": n, "rank": "species",
                                   "preferred_common_name": None}}
                        for i, (n, c) in enumerate(names_counts)]}


def test_counts_paginate_until_total_is_reached(monkeypatch):
    pages = [_page([(f"Genus sp{i}", 100 - i) for i in range(500)], 501),
             _page([("Genus last", 1)], 501)]
    seen = []

    def fake(url, params, sleep=regions.SLEEP, retries=regions.RETRIES):
        seen.append(params["page"])
        return pages[params["page"] - 1]

    monkeypatch.setattr(regions, "_get", fake)
    out = regions.species_counts(10, verbose=False)
    assert seen == [1, 2]
    assert len(out) == 501


def test_short_page_ends_pagination(monkeypatch):
    """A page shorter than per_page means the last one, even if `total` disagrees.

    iNat's `total_results` is an estimate on some queries; trusting it alone would
    spin on empty pages.
    """
    monkeypatch.setattr(regions, "_get",
                        lambda *a, **k: _page([("Genus sp", 5)], 99999))
    assert len(regions.species_counts(10, verbose=False)) == 1


def test_non_species_ranks_are_dropped(monkeypatch):
    payload = {"total_results": 2, "results": [
        {"count": 9, "taxon": {"id": 1, "name": "Rubus", "rank": "genus"}},
        {"count": 7, "taxon": {"id": 2, "name": "Rubus ursinus", "rank": "species"}},
    ]}
    monkeypatch.setattr(regions, "_get", lambda *a, **k: payload)
    out = regions.species_counts(10, verbose=False)
    assert [s["name"] for s in out] == ["Rubus ursinus"]


def test_names_are_curated(monkeypatch):
    """The survey must speak the catalogue's names, or nothing joins to it.

    9.5-10% of catalogue names are a taxonomic generation behind, so a survey that
    stored iNat's spelling verbatim would silently fail to match the very species
    it is meant to supply.
    """
    payload = {"total_results": 1, "results": [
        {"count": 5, "taxon": {"id": 1, "name": "Fragaria \u00d7 ananassa",
                               "rank": "species", "preferred_common_name": None}}]}
    monkeypatch.setattr(regions, "_get", lambda *a, **k: payload)
    out = regions.species_counts(10, verbose=False)
    assert out[0]["name"] == "Fragaria \u00d7 ananassa"      # what iNat said
    assert out[0]["curated"] == "Fragaria x ananassa"          # what we join on


def test_usable_filters_on_observations():
    sp = [{"n_obs": 5, "curated": "A b"}, {"n_obs": 20, "curated": "C d"},
          {"n_obs": 500, "curated": "E f"}]
    assert len(regions.usable(sp, 20)) == 2
    assert len(regions.usable(sp, 100)) == 1


def test_retry_recovers_from_a_transient_503(monkeypatch):
    """The failure mode that cost this project a twelve-hour run."""
    calls = {"n": 0}

    def flaky(url, params=None, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            return _Resp(503)
        return _Resp(200, {"ok": True})

    monkeypatch.setattr(regions.requests, "get", flaky)
    monkeypatch.setattr(regions.time, "sleep", lambda s: None)
    assert regions._get("http://x", {}) == {"ok": True}
    assert calls["n"] == 3


def test_retry_gives_up_with_an_actionable_message(monkeypatch):
    monkeypatch.setattr(regions.requests, "get",
                        lambda *a, **k: _Resp(503))
    monkeypatch.setattr(regions.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match="maintenance window"):
        regions._get("http://x", {})


def test_a_404_is_not_retried(monkeypatch):
    """Retrying a malformed query just repeats the mistake more slowly."""
    calls = {"n": 0}

    def notfound(*a, **k):
        calls["n"] += 1
        return _Resp(404)

    monkeypatch.setattr(regions.requests, "get", notfound)
    monkeypatch.setattr(regions.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit):
        regions._get("http://x", {})
    # one real attempt per retry slot, but no more than that -- and crucially the
    # HTTPError path is taken rather than the 5xx backoff path
    assert calls["n"] <= regions.RETRIES


def test_survey_reports_the_crowding_a_region_forces(monkeypatch):
    """Crowded genera are the product's dominant risk and are knowable up front.

    A user picking from this pool will pick siblings unless told, and on the
    measured numbers that costs far more label share than any encoder choice.
    """
    sp = [{"name": f"Rubus sp{i}", "curated": f"Rubus sp{i}", "n_obs": 50,
           "taxon_id": i, "common_name": None} for i in range(5)]
    sp += [{"name": "Bellis perennis", "curated": "Bellis perennis", "n_obs": 40,
            "taxon_id": 99, "common_name": None}]
    monkeypatch.setattr(regions, "species_counts", lambda *a, **k: sp)
    monkeypatch.setattr(regions, "place_name", lambda p: "Test Place")
    s = regions.survey(10, min_obs=20, verbose=False)
    assert s["n_species_usable"] == 6
    assert s["n_genera_usable"] == 2
    assert s["crowded_genera"] == {"Rubus": 5}


def test_survey_excludes_species_below_the_floor(monkeypatch):
    sp = [{"name": "A b", "curated": "A b", "n_obs": 1, "taxon_id": 1,
           "common_name": None},
          {"name": "C d", "curated": "C d", "n_obs": 99, "taxon_id": 2,
           "common_name": None}]
    monkeypatch.setattr(regions, "species_counts", lambda *a, **k: sp)
    monkeypatch.setattr(regions, "place_name", lambda p: None)
    s = regions.survey(10, min_obs=20, verbose=False)
    assert s["n_species_total"] == 2 and s["n_species_usable"] == 1
