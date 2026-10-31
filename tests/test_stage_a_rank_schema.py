"""The ranking response schema, its strict-order check and the ranking prompt."""

import json

import pytest
from pydantic import ValidationError

from pipeline.instrument import load_release
from pipeline.prompts import render_prompt
from pipeline.providers.gemini import gemini_response_schema
from pipeline.schemas.stage_a_rank import DimensionRanking, RankResponse, check_rank_response
from pipeline.stage_b.panel_books import load_panel_books

CODES = ("pitch_positivity", "expressed_enthusiasm", "interpersonal_warmth")
LABELS = ("A", "B", "C")
TEMPLATE = "stage_a_rank.j2"


def _response(**overrides) -> RankResponse:
    rankings = [{"code": code, "order": list(LABELS), "note": "A smiles throughout; C stays flat."}
                for code in CODES]
    return RankResponse.model_validate({"rankings": rankings, "notes": "", **overrides})


def _with_order(order) -> RankResponse:
    resp = _response()
    resp.rankings[1].order = list(order)
    return resp


def test_valid_response_has_no_problems():
    assert check_rank_response(_response(), CODES, LABELS) == []


def test_order_may_be_any_permutation_and_differ_by_dimension():
    assert check_rank_response(_with_order(["C", "A", "B"]), CODES, LABELS) == []


def test_missing_code_is_a_problem():
    resp = _response()
    del resp.rankings[2]
    problems = check_rank_response(resp, CODES, LABELS)
    assert len(problems) == 1 and "interpersonal_warmth" in problems[0] and "missing" in problems[0]


def test_duplicate_code_is_a_problem():
    resp = _response()
    resp.rankings.append(resp.rankings[0].model_copy())
    problems = check_rank_response(resp, CODES, LABELS)
    assert len(problems) == 1 and "pitch_positivity" in problems[0] and "more than once" in problems[0]


def test_extra_code_is_a_problem():
    resp = _response()
    resp.rankings.append(DimensionRanking(code="charisma", order=list(LABELS), note=""))
    problems = check_rank_response(resp, CODES, LABELS)
    assert len(problems) == 1 and "charisma" in problems[0] and "unknown" in problems[0]


def test_order_with_a_repeated_label_is_a_problem():
    problems = check_rank_response(_with_order(["A", "B", "B", "C"]), CODES, LABELS)
    assert len(problems) == 1 and "expressed_enthusiasm" in problems[0] and "'B'" in problems[0]


def test_order_missing_a_label_is_a_problem():
    problems = check_rank_response(_with_order(["A", "B"]), CODES, LABELS)
    assert len(problems) == 1 and "expressed_enthusiasm" in problems[0] and "'C'" in problems[0]


def test_order_with_an_unknown_label_is_a_problem():
    problems = check_rank_response(_with_order(["A", "B", "C", "D"]), CODES, LABELS)
    assert len(problems) == 1 and "expressed_enthusiasm" in problems[0] and "'D'" in problems[0]


def test_tied_clips_written_as_one_entry_are_a_problem():
    problems = check_rank_response(_with_order(["A", "B=C"]), CODES, LABELS)
    assert problems and all("expressed_enthusiasm" in p for p in problems)


def test_schema_is_strict_and_bounds_the_notes():
    with pytest.raises(ValidationError):
        _response(score=4)
    with pytest.raises(ValidationError):
        DimensionRanking(code="x", order=["A"], note="n" * 301)
    with pytest.raises(ValidationError):
        _response(notes="n" * 401)
    with pytest.raises(ValidationError):
        RankResponse.model_validate({"rankings": [{"code": "x", "order": ["A"], "note": "", "score": 3}],
                                     "notes": ""})


def test_wire_schema_is_provider_compatible():
    wire = gemini_response_schema(RankResponse)
    text = json.dumps(wire)
    for banned in ("$ref", "additionalProperties", "anyOf", "$defs"):
        assert banned not in text
    item = wire["properties"]["rankings"]["items"]
    assert set(item["properties"]) == {"code", "order", "note"}
    assert item["properties"]["order"] == {"items": {"type": "string"}, "type": "array"}
    assert set(wire["required"]) == {"rankings", "notes"}


def _prompt(labels=("A", "B", "C", "D", "E")):
    release = load_release()
    book = load_panel_books(release).stage_a
    return book, render_prompt(TEMPLATE, codebook=book, context={"labels": list(labels)},
                               prompt_dir=release.templates_dir)


def test_prompt_renders_all_three_dimensions_with_their_definitions():
    book, text = _prompt()
    assert book.ids == CODES
    for entry in book.entries:
        assert f"### {entry.id}\n" in text
        assert f"Definition: {entry.definition}" in text
        assert f"Include if: {entry.include_if}" in text
        assert f"Exclude if: {entry.exclude_if}" in text


def test_prompt_names_the_clip_labels_of_the_batch():
    _, text = _prompt()
    assert "The clips are labelled A, B, C, D, E." in text
    _, three = _prompt(("A", "B", "C"))
    assert "The clips are labelled A, B, C." in three and "D, E" not in three


def test_prompt_states_the_ranking_rules():
    _, text = _prompt()
    assert "No ties" in text and "Do not give scores" in text
    assert "strict order" in text and "prepared presentation" in text
    assert "do not guess what happened afterwards" in text
    flat = " ".join(text.split())
    assert ("2. For each dimension below, put the clips in a strict order from the one that shows the MOST of that "
            "dimension to the one that shows the LEAST. 3. No ties.") in flat
    assert "negative dimension" not in flat
    assert ("8. The definitions below were written for rating one presentation at a time. Where they mention a "
            "rating, a score or a number, read it as more or less of the dimension. Where a definition says to apply "
            "the shared protocol or to mark a clip uncodable, follow these rules instead.") in flat
    assert ("7. If missing or unusable founder speech or view of the presenters prevents the judgement on a "
            "dimension, still place the clip, last on that dimension, and say so in the note.") in flat
    assert text.index("7. If missing") < text.index("8. The definitions below") < text.index("DIMENSIONS")


def test_prompt_asks_for_no_score_anchor_or_prediction():
    book, text = _prompt()
    for entry in book.entries:
        for anchor in entry.anchors:
            assert anchor.description.strip()[:40] not in text
    # the only mention of investors is the statement that the clips stop before questioning
    lowered = " ".join(text.lower().split()).replace("cut before any investor questioning", "")
    # pitch_positivity's Exclude-if names funding only as something the judgement must not use
    assert lowered.count("funding") == lowered.count("or inferred funding success. do not derive") == 1
    lowered = lowered.replace("or inferred funding success. do not derive", "")
    for word in ("predict", "invest", "funding", "1-5", "1–5", "scale of"):
        assert word not in lowered


def test_provenance_never_reaches_the_ranking_prompt():
    book, text = _prompt()
    for entry in book.entries:
        assert entry.provenance, f"{entry.id} has no provenance to check"
        assert entry.provenance.strip()[:60] not in text
    for tool in ("Face++", "pyAudioAnalysis", "Loughran", "Nicolas", "LSTM", "Berlin"):
        assert tool not in text, f"{tool} leaked into the prompt"
