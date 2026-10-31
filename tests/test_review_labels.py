# tests/test_review_labels.py
import pytest
from pipeline.instrument import load_release
from pipeline.stage_b.panel_books import load_panel_books
from pipeline.stage_b import review_labels as rl
from tests.panel_fixtures import synthetic_record


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


def test_entry_for_resolves_codes_stage_a_topic_and_the_pitch_fields(books):
    assert rl.entry_for("turn:4/code:PR-SEV", books).id == "PR-SEV"
    assert rl.entry_for("stage_a:expressed_enthusiasm", books).id == "expressed_enthusiasm"
    assert rl.entry_for("exchange:1/topic", books).id == "EX-TOPIC"
    with pytest.raises(ValueError, match="no such path"):                       # no EX-CLOSE since v3-dev9
        rl.entry_for("exchange:1/investor:S2/closing", books)
    assert rl.entry_for("pitch:any_offer", books) is None                       # the outcome glosses are its definition
    assert rl.entry_for("pitch:first_decision", books).id == "DEC-KIND"
    assert rl.entry_for("prep_pres:PRES-EVID", books).id == "PRES-EVID"
    assert rl.entry_for("pair:b01/expressed_enthusiasm", books).id == "expressed_enthusiasm"
    assert rl.entry_for("venture:founder_gender", books).id == "VC-GENDER"
    assert rl.entry_for("venture:ask", books).id == "VC-ASK"


def test_default_question_text_uses_the_codebook_name_and_the_unit(books):
    d = synthetic_record()
    empty = rl.Labels({}, {}, {}, {})
    assert rl.question_text("turn:4/code:PR-SEV", d, books, empty) == "Question severity: turn 4 (Investor B)"
    assert rl.question_text("turn:4/exchange", d, books, empty) == \
        "turn:4/exchange: turn 4 (Investor B)"                                   # no codebook entry: the path
    assert rl.question_text("stage_a:expressed_enthusiasm", d, books, empty).startswith("Expressed enthusiasm")


def test_overrides_fill_placeholders(books):
    d = synthetic_record()
    labels = rl.Labels({"PR-SEV": "How hard does the question in turn {turn} press?",
                        "turn_function": "What does {who} do in {unit}?"}, {}, {}, {})
    assert rl.question_text("turn:4/code:PR-SEV", d, books, labels) == "How hard does the question in turn 4 press?"
    assert rl.question_text("turn:4/function", d, books, labels) == "What does Investor B do in turn 4 (Investor B)?"


def test_anchors_glosses_and_definition_html(books):
    sev = rl.entry_for("turn:4/code:PR-SEV", books)
    assert set(rl.scale_anchors(sev)) == {"1", "2", "3"} and "Understanding" in rl.scale_anchors(sev)["1"]
    function = rl.entry_for("turn:4/function", books)
    glosses = rl.value_glosses(function, ("accepts", "social", "repair"), rl.Labels({}, {}, {}, {}))
    assert glosses["accepts"].startswith("an investor explicitly shows acceptance")
    html = rl.definition_html(rl.entry_for("turn:4/code:PR-CONCEPT", books))
    assert html.count("<p>") > 10 and "<b>customer acquisition</b>" in html
    assert rl.scale_anchors(None) == {} and rl.value_glosses(None, ("a",), rl.Labels({}, {}, {}, {})) == {}
    gender = rl.entry_for("venture:founder_gender", books)           # categorical, no anchors, no VALUE: glosses
    assert rl.scale_anchors(gender) == {} and rl.definition_html(gender).startswith("<p>")
    assert rl.definition_html(None) == ""


def test_unit_kind_section_and_structural_text():
    labels = rl.Labels({}, {"turn": "this turn is not what it says"}, {}, {})
    assert rl.unit_kind("turn:4/code:PR-SEV") == "turn" and rl.section_of("turn:4/code:PR-SEV") == "conversation"
    assert rl.unit_kind("exchange:1/topic") == "exchange"
    assert rl.unit_kind("stage_a:expressed_enthusiasm") == "pitch" and rl.section_of("stage_a:expressed_enthusiasm") == "pitch"
    assert rl.section_of("prep_pres:PRES-SUBS") == rl.section_of("venture:ask") == "pitch"
    for path in ("pitch:any_offer", "pitch:first_decision"):
        assert rl.unit_kind(path) == "actor" and rl.section_of(path) == "outcomes"
    assert rl.structural_text("turn:4/code:PR-SEV", labels) == "this turn is not what it says"
    assert rl.structural_text("exchange:1/topic", labels) == rl.DEFAULT_STRUCTURAL["exchange"]


def test_the_release_labels_file_loads_and_covers_the_current_codes():
    labels = rl.load_labels(load_release())
    assert "PR-SEV" in labels.questions and "outcome" in labels.questions and "turn" in labels.structural


def test_the_release_labels_word_every_pitch_level_card(books):
    d = synthetic_record()
    labels = rl.load_labels(load_release())
    for code in books.prepared_codes():
        assert rl.entry_for(f"prep_pres:{code.id}", books).id == code.id
        assert code.id in labels.questions, code.id
    # the coders' wording since Task 7d (the researcher's pilot coding); "mark now" is in the card's line below it
    assert rl.question_text("pitch:first_decision", d, books, labels) == \
        "When does the first shark say they are out or make an offer?"
    assert "mark now" in labels.hints["cut_card"] and "even if others keep asking" in labels.hints["cut_card"]
    assert rl.question_text("turn:4/code:RS-EVID", d, books, labels) == \
        "Does the answer in turn 4 give at least one checkable specific?"
    assert rl.question_text("venture:ask", d, books, labels) == \
        "What did the founders ask for, and for what share? (dollars;per cent)"
    assert rl.question_text("prep_pres:PRES-SUBS", d, books, labels) == \
        "How much does the presentation explain about how the business works?"
    # a pair row asks its own item's question (plan C Task 7e: one comparison task per pair)
    assert rl.question_text("pair:b01/expressed_enthusiasm", d, books, labels) == \
        "Which pitch comes across as more enthusiastic?"
    assert rl.question_text("pair:b01/pitch_positivity", d, books, labels) == "Which pitch comes across as more positive?"
    assert rl.question_text("pair:b01/interpersonal_warmth", d, books, labels) == "Which pitch comes across as warmer?"
    # the Stage A card of the same item keeps its own question
    assert "comes across" not in rl.question_text("stage_a:expressed_enthusiasm", d, books, labels)
    assert "exchange_investor" not in labels.structural and "exchange_investor" not in rl.DEFAULT_STRUCTURAL
    assert "closing" not in labels.questions
    assert set(labels.glosses["outcome"]) == {"yes", "no", "unknown"} and "DEC-KIND" not in labels.glosses
    assert "never seen deciding" in labels.glosses["outcome"]["no"]
    assert "before the panel decides" in labels.glosses["outcome"]["unknown"]
    for key in ("prep_pres_card", "cut_card", "pairs_intro", "pitch_intro"):
        assert labels.hints[key] and key in {**rl.DEFAULT_HINTS, **labels.hints}
    assert "‘mark now’" in labels.hints["cut_card"] and "‘close’" in labels.hints["pairs_intro"]


def test_the_pitch_intro_hint_is_a_paragraph_of_the_shared_protocol():
    release = load_release()
    labels = rl.load_labels(release)
    protocol = " ".join((release.templates_dir / "stage_a_perceived_delivery_protocol.md").read_text().split())
    assert " ".join(labels.hints["pitch_intro"].split()) in protocol
    assert labels.hints["pitch_intro"].startswith("Watch the entire supplied clip")


def test_the_outcome_glosses_are_what_a_card_shows_as_its_definition(books):
    labels = rl.load_labels(load_release())
    g = rl.path_glosses("pitch:any_offer", None, ("yes", "no", "unknown"), labels)
    assert g == labels.glosses["outcome"] and set(g) == {"yes", "no", "unknown"}
    assert rl.path_glosses("pitch:any_offer", None, ("yes",), labels) == {"yes": labels.glosses["outcome"]["yes"]}
    sev = rl.entry_for("turn:4/code:PR-SEV", books)
    assert rl.path_glosses("turn:4/code:PR-SEV", sev, ("1",), labels) == rl.value_glosses(sev, ("1",), labels)


def test_every_concept_value_has_a_gloss_from_the_definitions(books):
    empty = rl.Labels()
    for code in ("PR-CONCEPT", "RS-CONCEPT", "EX-TOPIC"):
        entry = books.panel.by_id(code)
        glosses = rl.value_glosses(entry, tuple(entry.values), empty, books.panel)
        assert len(entry.values) == 19 and set(glosses) == set(entry.values), code
        assert all(len(g) > 3 and not g.endswith(":") for g in glosses.values()), code
    pr = rl.value_glosses(books.panel.by_id("PR-CONCEPT"), tuple(books.panel.by_id("PR-CONCEPT").values), empty, books.panel)
    assert pr["customer_acquisition"] == "how customers are won, obtained or onboarded"
    assert pr["execution"].startswith("whether it works") and "Prevention" not in pr["forecast"]
    assert pr["deal_terms"].startswith("the ask, the valuation") and "Deal terms" not in pr["team_capabilities"]
    assert pr["other"].startswith("the question is about a real business matter") and pr["NONE"].startswith("the question only repeats")
    rs = rl.value_glosses(books.panel.by_id("RS-CONCEPT"), tuple(books.panel.by_id("RS-CONCEPT").values), empty, books.panel)
    assert rs["sales"] == pr["sales"] and rs["NONE"].startswith("a bare acknowledgment") and rs["other"].startswith("a real business matter")
    ex = rl.value_glosses(books.panel.by_id("EX-TOPIC"), tuple(books.panel.by_id("EX-TOPIC").values), empty, books.panel)
    assert ex["vision"] == pr["vision"] and ex["NONE"].startswith("the exchange is social")
    # the labels' own glosses win over the parsed ones, and the borrowing is by value, not by name
    won = rl.value_glosses(books.panel.by_id("RS-CONCEPT"), ("sales", "vision"), rl.Labels(glosses={"RS-CONCEPT": {"sales": "x"}}), books.panel)
    assert won == {"sales": "x", "vision": pr["vision"]}
    assert rl.value_glosses(books.panel.by_id("RS-CONCEPT"), ("sales",), empty) == {}       # no book to borrow from


def test_the_new_units_have_a_kind_a_section_and_a_structural_text():
    assert rl.DEFAULT_SECTIONS["clips"] == "Prepared presentation (clip session)"
    assert rl.DEFAULT_SECTIONS["pairs"] == "Which shows more?"
    assert rl.unit_kind("pair:b01/pitch_positivity") == "pair" and rl.section_of("pair:b01/pitch_positivity") == "pairs"
    assert rl.structural_text("pair:b01/pitch_positivity", rl.Labels())
    assert rl.unit_kind("prep_pres:PRES-SUBS") == rl.unit_kind("venture:ask") == "pitch"


def test_outcome_question_defaults_to_the_decision_entry_name_and_overrides_win(books):
    d = synthetic_record()
    assert rl.question_text("pitch:any_offer", d, books, rl.Labels()) == "Outcome: the whole video"
    assert rl.question_text("pitch:any_offer", d, books,
                            rl.Labels({"outcome": "Did any investor offer in {unit}?"})) == \
        "Did any investor offer in the whole video?"
    assert rl.question_text("pitch:any_offer", d, books, rl.load_labels(load_release())) == \
        "Did any investor make or join an offer at any point?"


def test_structure_entry_resolves_the_four_roles_and_nothing_else(books):
    from pipeline.stage_b.panel_books import STRUCTURE_CODES
    for role, code_id in STRUCTURE_CODES.items():
        assert books.structure_entry(role).id == code_id
    assert len(STRUCTURE_CODES) == 4 and books.structure_entry("no_such_role") is None
    assert books.structure_entry("closing") is None                            # no closing state since v3-dev9


def test_a_book_missing_ids_gives_no_entry_instead_of_raising(books):
    from dataclasses import replace
    from pipeline.codebook import Codebook
    d = synthetic_record()
    kept = [e for e in books.panel.entries if e.id != "PR-SEV"]
    thin = replace(books, panel=Codebook(entries=kept),
                   stage_a=Codebook(entries=[]), venture=Codebook(entries=[]))
    for path in ("turn:4/code:PR-SEV", "pair:b01/expressed_enthusiasm", "stage_a:expressed_enthusiasm",
                 "venture:founder_gender"):
        assert rl.entry_for(path, thin) is None
        text = rl.question_text(path, d, thin, rl.Labels())
        assert isinstance(text, str) and path in text
    assert rl.question_text("turn:4/code:PR-SEV", d, thin, rl.Labels()).endswith("turn 4 (Investor B)")
    assert rl.entry_for("exchange:1/topic", thin).id == "EX-TOPIC"
    assert rl.scale_anchors(None) == {} and rl.definition_html(None) == ""


def test_only_the_kind_of_a_decision_has_an_entry(books):
    assert rl.entry_for("decision:1/kind", books).id == books.structure_entry("decision").id
    for field in ("time", "actor", "offer_id"):
        assert rl.entry_for(f"decision:1/{field}", books) is None


def test_definition_html_escapes_the_definition_and_the_anchors(books):
    import re
    sev = rl.entry_for("turn:4/code:PR-SEV", books)
    anchors = [sev.anchors[0].model_copy(update={"description": "< 5 minutes & <script>"}), *sev.anchors[1:]]
    nasty = sev.model_copy(update={"definition": "a < b & c > d", "anchors": anchors})
    out = rl.definition_html(nasty)
    assert "&lt;" in out and "&gt;" in out and "&amp;" in out
    assert "&lt;script&gt;" in out and "a &lt; b &amp; c &gt; d" in out
    assert set(re.findall(r"<[^>]*>", out)) <= {"<p>", "</p>", "<b>", "</b>"}
    assert out.count("<p><b>") == len(sev.anchors)


# ---- the "cannot be rated" answers, the unit's words and a labels file that cannot be used ----------

def test_missing_options_are_the_reasons_the_import_takes_for_a_turn_code_and_for_nothing_else():
    from pipeline.schemas.stage_b_panel import MISSING_REASONS
    options = rl.missing_options("turn:5/code:PREP-SUBS", rl.Labels())
    assert [o["value"] for o in options] == [f"missing:{r}" for r in MISSING_REASONS if r != "none"]
    assert "missing:none" not in [o["value"] for o in options]
    by_value = {o["value"]: o["label"] for o in options}
    assert by_value["missing:too_short"] == "too short" and by_value["missing:not_applicable"] == "not applicable"
    assert rl.missing_options("turn:4/code:PR-CONCEPT", rl.Labels()) == options       # every turn code, of any type
    worded = rl.missing_options("turn:5/code:PREP-SUBS", rl.Labels(missing_reasons={"too_short": "over too quickly"}))
    assert {o["value"]: o["label"] for o in worded}["missing:too_short"] == "over too quickly"
    assert {o["value"]: o["label"] for o in worded}["missing:ambiguous"] == "ambiguous"
    # the import takes no missing:<reason> for these, so their cards offer none
    for path in ("stage_a:expressed_enthusiasm", "exchange:1/topic", "pitch:any_offer", "pitch:first_decision",
                 "venture:founder_gender", "turn:4/function", "prepared_window"):
        assert rl.missing_options(path, rl.Labels()) == []


def test_the_unit_of_a_default_question_is_worded_by_the_labels_layer(books):
    d = synthetic_record()
    assert rl._unit_text("exchange:1/topic", d)["unit"] == f"{rl.DEFAULT_EXCHANGE_UNIT.lower()} 1"
    assert rl._unit_text("stage_a:expressed_enthusiasm", d)["unit"] == f"the {rl.DEFAULT_SECTIONS['pitch'].lower()}"
    labels = rl.Labels(sections={"exchange_unit": "Thread", "pitch": "Opening Talk"})
    assert rl._unit_text("exchange:1/topic", d, labels)["unit"] == "thread 1"
    assert rl.question_text("stage_a:expressed_enthusiasm", d, books, labels).endswith(": the opening talk")
    asked = rl.Labels(questions={"stage_a": "Rate {unit}."}, sections={"pitch": "Opening Talk"})
    assert rl.question_text("stage_a:expressed_enthusiasm", d, books, asked) == "Rate the opening talk."


def _release_with_labels(tmp_path, text):
    import dataclasses
    path = tmp_path / "review_labels.yaml"
    path.write_text(text, encoding="utf-8")
    return dataclasses.replace(load_release(), review_labels=path)


@pytest.mark.parametrize("text, named", [
    ("- questions\n- structural\n", "must be a mapping"),                                   # the root is a list
    ("questions: {}\nquestons: {PR-SEV: x}\n", "'questons'"),                               # an unknown top-level key
    ("questions: just some text\n", "`questions` must be a mapping"),                       # a section that is no mapping
    ("glosses: [a, b]\n", "`glosses` must be a mapping"),
    ("questions:\n  PR-SEV: 'How hard does {speaker} press in turn {turn}?'\n", "'PR-SEV'")])    # unknown placeholder
def test_a_labels_file_that_cannot_be_used_is_refused_naming_what_is_wrong(tmp_path, text, named):
    from pipeline.errors import ConfigError
    with pytest.raises(ConfigError) as caught:
        rl.load_labels(_release_with_labels(tmp_path, text))
    assert named in str(caught.value) and "review_labels" in str(caught.value)
    if "speaker" in text:
        assert "'speaker'" in str(caught.value) and "{who}, {turn}, {unit}" in str(caught.value)
    if "questons" in text:
        assert "missing_reasons" in str(caught.value)             # the keys the file may have are listed


def test_a_labels_file_may_be_empty_or_partial_and_may_word_the_missing_reasons(tmp_path):
    assert rl.load_labels(_release_with_labels(tmp_path, "")) == rl.Labels()
    labels = rl.load_labels(_release_with_labels(tmp_path, "missing_reasons:\n  too_short: over too quickly\nhints:\n"))
    assert labels.missing_reasons == {"too_short": "over too quickly"} and labels.hints == {} and labels.questions == {}


def test_a_question_with_an_unknown_placeholder_is_refused_when_it_is_filled(books):
    from pipeline.errors import ConfigError
    labels = rl.Labels(questions={"PR-SEV": "How hard does {speaker} press?"})
    with pytest.raises(ConfigError, match=r"'PR-SEV'.*\{who\}, \{turn\}, \{unit\}"):
        rl.question_text("turn:4/code:PR-SEV", synthetic_record(), books, labels)


# ---- the coders' own wording of a code (plan C, Task 7d) --------------------------------------------------------

@pytest.mark.parametrize("text, named", [
    ("anchors:\n  NO-SUCH:\n    gloss: x\n", "'NO-SUCH'"),                                   # a code the books lack
    ("anchors:\n  PR-SEV:\n    values:\n      4: too far\n", "no value '4'"),                # beyond PR-SEV's 1-3
    ("anchors:\n  RS-EVID:\n    values:\n      maybe: x\n", "no value 'maybe'"),
    ("anchors:\n  PR-CONCEPT:\n    values:\n      franchising: x\n", "no value 'franchising'"),
    ("anchors:\n  PR-SEV:\n    glos: x\n", "mapping of gloss and values"),                  # an unknown key
    ("anchors:\n  PR-SEV:\n    values:\n      1: {lable: x}\n", "mapping of label and text"),
    ("anchors:\n  PR-SEV:\n    values:\n      1: 3\n", "mapping of label and text"),        # wording that is no text
    ("anchors:\n  PR-SEV: just text\n", "mapping of gloss and values"),
    ("anchors: [PR-SEV]\n", "`anchors` must be a mapping")])
def test_a_coders_wording_for_an_unknown_code_or_value_is_refused(tmp_path, text, named):
    from pipeline.errors import ConfigError
    with pytest.raises(ConfigError) as caught:
        rl.load_labels(_release_with_labels(tmp_path, text))
    assert named in str(caught.value) and "review_labels" in str(caught.value)


def test_a_coders_wording_is_read_in_one_shape(tmp_path):
    labels = rl.load_labels(_release_with_labels(tmp_path, (
        "anchors:\n  RS-EVID:\n    values:\n      true: {label: 'yes'}\n      'false': {label: 'no', text: none given}\n"
        "  PR-SEV:\n    gloss: judge what it does\n    values:\n      1: asks\n")))
    assert labels.anchors == {
        "RS-EVID": {"gloss": "", "values": {"true": {"label": "yes", "text": None},
                                            "false": {"label": "no", "text": "none given"}}},
        "PR-SEV": {"gloss": "judge what it does", "values": {"1": {"label": None, "text": "asks"}}}}
    assert rl.load_labels(_release_with_labels(tmp_path, "anchors:\n")).anchors == {}


def test_the_release_words_six_codes_for_the_coders(books):
    labels = rl.load_labels(load_release())
    assert set(labels.anchors) == {"RS-EVID", "PR-SEV", "PREP-SUBS", "PRES-SUBS", "PRES-COH", "PREP-COH"}
    assert labels.anchors["PREP-SUBS"] == labels.anchors["PRES-SUBS"]
    assert labels.questions["PR-SEV"] == "How hard does the question in turn {turn} push on what the founder has said?"
    assert labels.questions["PREP-SUBS"] == "How much does the answer in turn {turn} explain about how the business works?"
    assert labels.questions["RS-EVID"] == "Does the answer in turn {turn} give at least one checkable specific?"
    assert labels.sections["first_decision"] == "The first decision"
    assert labels.questions["PRES-COH"] == "Does the presentation’s story hang together?"
    assert labels.anchors["PR-SEV"]["values"]["3"]["text"].startswith("Pushes back:")
    assert labels.anchors["PRES-COH"]["values"]["5"]["text"].startswith("Clear and answers the obvious doubt")
    assert labels.anchors["PR-SEV"]["gloss"].endswith("Judge what the question does, not its tone.")
    # the coders' wording leaves the codebook as it is: the model still reads its own scale
    assert rl.scale_anchors(books.panel.by_id("PR-SEV"))["1"].startswith("Understanding-seeking")
    entry = books.panel.by_id("RS-EVID")
    assert rl.value_labels(entry, labels) == {"true": "yes", "false": "no"} and rl.coder_anchors(entry, labels) == {}
    assert rl.question_gloss(books.panel.by_id("RS-RESP"), labels) == "" and rl.coder_anchors(None, labels) == {}


def test_answer_coherence_is_worded_for_the_coders(books):
    """Plan C Task 7e (researcher, 2026-10-21): answer coherence, the strongest single feature on v3-dev9, joins the
    question-answer codes with the coders' own wording; the codebook keeps Gemini's."""
    labels = rl.load_labels(load_release())
    d = synthetic_record()
    assert rl.question_text("turn:4/code:PREP-COH", d, books, labels) == "Does the answer in turn 4 hang together?"
    entry = books.panel.by_id("PREP-COH")
    assert rl.question_gloss(entry, labels) == ("Do its points follow from one another, with no step missing and "
                                                "nothing that contradicts itself? Do not count nerves, accents or "
                                                "stumbles.")
    assert rl.coder_anchors(entry, labels) == {
        "1": "Doesn't connect: points jump around or contradict each other.",
        "2": "Missing steps: a conclusion with the reasons missing.",
        "3": "Mostly clear: one step skipped or unsupported.",
        "4": "Clear throughout: each point follows.",
        "5": "Clear and answers the obvious doubt before it is raised."}
    assert rl.scale_anchors(entry)["1"].startswith("Incoherent.")         # the model's scale is unchanged
