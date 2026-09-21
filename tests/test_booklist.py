"""A field guide's list, resolved — and the ambiguity it must refuse to resolve."""

import json
import pytest

from plantid.data import booklist as B

SURVEY = [
    {"name": "Gaultheria shallon", "curated": "Gaultheria shallon",
     "common_name": "salal", "n_obs": 5000},
    {"name": "Camassia quamash", "curated": "Camassia quamash",
     "common_name": "camash", "n_obs": 2000},
    {"name": "Camassia leichtlinii", "curated": "Camassia leichtlinii",
     "common_name": "large camas", "n_obs": 2700},
    {"name": "Toxicoscordion venenosum", "curated": "Toxicoscordion venenosum",
     "common_name": "Watson's death camas", "n_obs": 685},
    {"name": "Conium maculatum", "curated": "Conium maculatum",
     "common_name": "poison hemlock", "n_obs": 900},
]


def r(names):
    return B.resolve(names, SURVEY, use_inat=False)


def test_a_common_name_resolves_to_its_species():
    assert r(["salal"])["resolved"] == {"salal": "Gaultheria shallon"}


def test_a_scientific_name_passes_through():
    assert r(["Conium maculatum"])["resolved"]["Conium maculatum"] == "Conium maculatum"


def test_case_and_punctuation_do_not_matter():
    assert r(["SALAL", "Salal."])["resolved"] == {
        "SALAL": "Gaultheria shallon", "Salal.": "Gaultheria shallon"}


def test_an_ambiguous_name_is_reported_and_never_guessed():
    """The case that matters most, and it is a safety case. 'camas' matches the
    edible Camassia AND Toxicoscordion — death camas — whose bulbs look similar
    and are lethal. Picking the most-photographed one silently would put a label
    in the catalogue that does not mean what the book meant."""
    out = r(["camas"])
    assert out["resolved"] == {}
    cands = {c["species"] for c in out["ambiguous"]["camas"]}
    assert "Camassia leichtlinii" in cands
    assert "Toxicoscordion venenosum" in cands, "death camas must be surfaced"


def test_a_dominant_match_is_not_treated_as_ambiguous():
    """Refusing everything with more than one candidate would make the tool
    useless. A clear winner wins."""
    survey = SURVEY + [{"name": "Gaultheria humifusa", "curated": "Gaultheria humifusa",
                        "common_name": "alpine salal", "n_obs": 12}]
    out = B.resolve(["salal"], survey, use_inat=False)
    assert out["resolved"] == {"salal": "Gaultheria shallon"}


def test_a_plant_from_elsewhere_is_not_silently_dropped():
    out = r(["Carnegiea gigantea"])
    assert out["not_in_region"] == ["Carnegiea gigantea"]
    assert out["resolved"] == {}


def test_nonsense_is_reported_as_not_found():
    assert r(["wibble plant"])["not_found"] == ["wibble plant"]


def test_comments_and_annotations_are_stripped(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("# a page\n\nsalal (evergreen shrub)\nCamassia quamash  # spring\n")
    assert B.read_names(f) == ["salal", "Camassia quamash"]
