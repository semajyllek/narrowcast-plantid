"""Regional fetch: licence parsing, cluster identity, and the redistribution gate.

Offline. The licence test exists because the first implementation read the last
path segment of a GBIF licence URL, which is always "legalcode", and therefore
classified every image as unknown and filtered all of them out -- silently, as an
empty result rather than an error.
"""
import pytest

from plantid.data import regional_fetch as rf


@pytest.mark.parametrize("url,expected", [
    ("http://creativecommons.org/licenses/by-nc/4.0/legalcode", "CC_BY_NC"),
    ("http://creativecommons.org/licenses/by/4.0/legalcode", "CC_BY"),
    ("http://creativecommons.org/licenses/by-sa/4.0/legalcode", "CC_BY_SA"),
    ("http://creativecommons.org/publicdomain/zero/1.0/legalcode", "CC0"),
    ("https://creativecommons.org/licenses/by-nc-nd/4.0/", "CC_BY_NC_ND"),
    (None, "UNKNOWN"),
    ("", "UNKNOWN"),
    ("http://example.org/whatever", "UNKNOWN"),
])
def test_licence_urls_parse_to_comparable_tokens(url, expected):
    assert rf.parse_licence(url) == expected


def test_legalcode_suffix_does_not_swallow_the_licence():
    """The exact bug: every GBIF licence URL ends /legalcode."""
    assert rf.parse_licence(
        "http://creativecommons.org/licenses/by-nc/4.0/legalcode") != "LEGALCODE"


def _rec(key, licence, n_media=2, species="Trillium ovatum"):
    return {"key": key, "species": species, "license": licence,
            "institutionCode": "X", "recordedBy": "Y",
            "media": [{"identifier": f"http://img/{key}_{i}.jpg",
                       "format": "image/jpeg"} for i in range(n_media)]}


def test_photos_of_one_plant_share_a_cluster(monkeypatch):
    """The trap the whole plan flags: several photos of one individual are not
    independent observations, and must not straddle a train/test split."""
    monkeypatch.setattr(rf, "_get", lambda *a, **k: {"results": [
        _rec("111", "http://creativecommons.org/licenses/by/4.0/legalcode", 3),
        _rec("222", "http://creativecommons.org/licenses/by/4.0/legalcode", 2),
    ]})
    out = rf.occurrences("Trillium ovatum", want=10, state_province="Oregon")
    assert len(out) == 5                      # photographs
    assert len({r["cluster"] for r in out}) == 2   # plants
    assert {r["cluster"] for r in out} == {"111", "222"}


def test_non_commercial_is_excluded_by_default(monkeypatch):
    monkeypatch.setattr(rf, "_get", lambda *a, **k: {"results": [
        _rec("1", "http://creativecommons.org/licenses/by-nc/4.0/legalcode"),
        _rec("2", "http://creativecommons.org/licenses/by/4.0/legalcode"),
    ]})
    strict = rf.occurrences("Trillium ovatum", want=10, state_province="Oregon")
    assert {r["cluster"] for r in strict} == {"2"}
    loose = rf.occurrences("Trillium ovatum", want=10, state_province="Oregon",
                           open_only=False)
    assert {r["cluster"] for r in loose} == {"1", "2"}
    # recorded either way, so the decision stays visible
    assert {r["licence"] for r in loose} == {"CC_BY_NC", "CC_BY"}


def test_a_synonym_record_is_dropped_not_relabelled(monkeypatch):
    """GBIF's scientificName search admits synonyms; a record filed under another
    name is a label mismatch, not a hard example."""
    monkeypatch.setattr(rf, "_get", lambda *a, **k: {"results": [
        _rec("1", "http://creativecommons.org/licenses/by/4.0/legalcode",
             species="Trillium albidum"),
        _rec("2", "http://creativecommons.org/licenses/by/4.0/legalcode"),
    ]})
    out = rf.occurrences("Trillium ovatum", want=10)
    assert {r["cluster"] for r in out} == {"2"}


def test_want_counts_plants_not_photographs(monkeypatch):
    """`--per-species` is occurrences, because that is the independent unit."""
    monkeypatch.setattr(rf, "_get", lambda *a, **k: {"results": [
        _rec(str(i), "http://creativecommons.org/licenses/by/4.0/legalcode", 4)
        for i in range(10)]})
    out = rf.occurrences("Trillium ovatum", want=3)
    assert len({r["cluster"] for r in out}) == 3
    assert len(out) == 12          # 3 plants x 4 photos


def test_non_image_media_is_skipped(monkeypatch):
    rec = _rec("1", "http://creativecommons.org/licenses/by/4.0/legalcode", 0)
    rec["media"] = [{"identifier": "http://x/a.mp3", "format": "audio/mpeg"},
                    {"identifier": "http://x/b.jpg", "format": "image/jpeg"}]
    monkeypatch.setattr(rf, "_get", lambda *a, **k: {"results": [rec]})
    out = rf.occurrences("Trillium ovatum", want=5)
    assert len(out) == 1 and out[0]["url"].endswith(".jpg")
