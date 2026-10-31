from pipeline.instrument import check_release, load_release
from pipeline.prompts import prompt_hash
from pipeline.stage_b.panel_books import PANEL_TEMPLATE, load_panel_books, render_panel_prompt


def _books_and_prompt():
    release = load_release()
    books = load_panel_books(release)
    return release, books, render_panel_prompt(books, release.templates_dir)


def test_release_lists_three_books_and_is_ready():
    release = load_release()
    assert set(release.books) == {"stage_a", "stage_b_panel", "venture_context"}
    assert release.books["stage_b_panel"].template == PANEL_TEMPLATE
    assert check_release(release) == []


def test_book_lookups():
    _, books, _ = _books_and_prompt()
    assert not hasattr(books, "event_kinds")
    functions = books.turn_functions()
    assert functions == ("presents", "question", "answer", "accepts", "challenges", "social", "hostile", "repair",
                         "demonstrates", "negotiates", "decides", "NONE")
    assert books.exchange_kinds() == ("substantive", "social", "demonstration", "negotiation")
    assert "operating_efficiency" in books.exchange_topics() and "NONE" in books.exchange_topics()
    assert not hasattr(books, "closing_states")
    assert "joins_offer" in books.decision_kinds() and "NONE" not in books.decision_kinds()
    assert [e.id for e in books.codes_for("question")] == ["PR-CONCEPT", "PR-SEV"]
    assert [e.id for e in books.codes_for("answer")] == ["RS-RESP", "RS-CONCEPT", "RS-EVID", "RS-UNCERT", "PREP-SUBS", "PREP-COH", "PREP-CTX"]
    assert books.codes_for("accepts") == () and books.codes_for("NONE") == ()
    assert books.stage_a_ids() == ("pitch_positivity", "expressed_enthusiasm", "interpersonal_warmth")
    assert "NONE" in books.venture_values("VC-INDUSTRY")


def test_prompt_renders_every_code_and_no_provenance():
    _, books, prompt = _books_and_prompt()
    for book in (books.panel, books.stage_a, books.venture):
        for entry in book.entries:
            assert entry.id in prompt
            assert entry.definition[:40] in prompt
            assert entry.include_if[:30] in prompt and entry.exclude_if[:30] in prompt
            if entry.provenance:
                assert entry.provenance[:50] not in prompt
            for anchor in entry.anchors:
                assert anchor.description[:30] in prompt
    for marker in ("Schegloff", "Kanze", "Hu & Ma", "et al", "Face++"):
        assert marker not in prompt


def test_prompt_states_the_documentation_rules():
    _, _, prompt = _books_and_prompt()
    lowered = prompt.lower()
    for phrase in ("entire video", "do not predict", "after any investor", "clock time",
                   "prepared presentation", "missing", "no quotes", "never drop the colon", "exchange",
                   "do not invent structure"):
        assert phrase in lowered
    assert "events —" not in prompt and "EV-KIND" not in prompt and "link_basis" not in prompt
    assert "friendly_reciprocation" not in prompt and "evidence_s" not in prompt
    assert "probability" not in lowered
    # the rules and record-field text is ours; the codebook text below it is rendered from the canonical books
    assert "source second" not in lowered[:lowered.index("codebook — panel interaction")]


def test_prompt_is_deterministic_and_has_no_per_video_content():
    release, books, prompt = _books_and_prompt()
    assert prompt_hash(prompt) == prompt_hash(render_panel_prompt(books, release.templates_dir))
    assert len(prompt) < 60000


def _flat(text):
    return " ".join(text.split())


def test_release_id_is_v3_dev9():
    assert load_release().release_id == "v3-dev9"


def test_prompt_states_the_v3_dev4_rules_verbatim():
    _, _, prompt = _books_and_prompt()
    flat = _flat(prompt)
    for phrase in (
        "7. Rate each turn on its own evidence. A later reaction, exit or offer must not change the rating of an "
        "earlier question or answer.",
        "8. Do not invent structure. Group turns into an exchange only when they are about the same matter, and "
        "name the turn a reply responds to only when the footage supports it. Presence in the room is not "
        "attention, and silence is not agreement.",
        "judged on the presenting team's displayed delivery across the prepared presentation spans only, "
        "following the shared protocol under STAGE A, DELIVERY OF THE PREPARED PRESENTATION below.",
        "For moment, give the clock time of the single moment that best shows the rating when there is one; null "
        "when the rating describes the presentation as a whole. Do not invent or space out times. If there is no "
        "prepared presentation, mark the item uncodable with no value.",
        "Both audible founder speech and a usable view of the presenters must be available for a meaningful portion "
        "of the prepared presentation; when missing or unusable material prevents the judgement, mark the item "
        "uncodable with no value and say why in the note. A low level of a construct is never missing evidence.",
        "Each item: a whole number 1 to 5 from the anchors below.",
        "covering the prepared presentation, the questioning, social talk, demonstrations and the negotiation.",
        "A clearly attributable non-verbal act that matters (shared laughter, a visible reaction of confusion) "
        "gets its own short turn for that actor.",
        "its function, as defined under TURN-FUNC below; who it addresses (actor ids, or PANEL, or UNKNOWN; empty "
        "for the prepared presentation); the exchange it belongs to, or null; the ids of earlier turns it responds "
        "to; its codes; other_concept; and a neutral summary of about 20 words.",
        "responds_to: an answer names the question or challenge turn it answers. An accepts, challenges or "
        "repair turn names the earlier turn it reacts to. Other turns leave it empty.",
        "codes: a question turn carries exactly PR-CONCEPT and PR-SEV. An answer turn carries exactly RS-RESP, "
        "RS-CONCEPT, RS-EVID, RS-UNCERT, PREP-SUBS, PREP-COH and PREP-CTX. Every other function carries an empty "
        "codes list.",
        "exchanges — the back-and-forth, as defined under EX-KIND below, numbered from 1 in order of first turn.",
        "when the panel returns to an earlier topic, give the later turns the earlier exchange's id.",
        "claim_conflicts — as defined under PREP-CONSIST below: the two turns holding the conflicting claims, the "
        "metric, a note, whether it was reconciled, and the turn that reconciled it (otherwise null).",
        "Write category values exactly as the codebook spells them, including capitals (NEUTRAL, NONE).",
        "An exit has a null offer_id.",
        "the turn in which the decision is spoken",
        "4. All times are clock times from the start of this video file, written M:SS: 0:07 for seven seconds, "
        "1:48 for one minute forty-eight seconds, 12:05 for twelve minutes five seconds. Never write a time as a "
        "plain number of seconds, and never drop the colon.",
        "the earliest clock time that supports the classification",
        "the clock time at which the decisive words begin",
        "Required on every `question` turn.",
        "Required on every `answer` turn.",
    ):
        assert phrase in flat, phrase


BRIDGE = ("In this record the protocol's words map as follows. Where the protocol says evidence, write the supporting "
          "behaviour in the item's note (about 15 words; the note may cite clock times of contrasting moments) and put "
          "the clock time of the single best-supporting moment in moment (null when the rating describes the whole "
          "presentation). Where it says overall_notes, use the "
          "record's notes. All times are clock times of this video file (M:SS from the start of the file), never "
          "clip-relative. The protocol's \"clip\" is the prepared presentation spans of this video.")


def test_the_stage_a_protocol_is_mapped_onto_the_record_fields():
    release, _, prompt = _books_and_prompt()
    protocol = (release.templates_dir / "stage_a_perceived_delivery_protocol.md").read_text().strip()
    assert protocol in prompt
    after = prompt[prompt.index(protocol) + len(protocol):].lstrip()
    assert " ".join(after.split()).startswith(BRIDGE)                       # directly after the protocol
    # overall_notes is named only by the protocol and by the paragraph that maps it onto `notes`
    rest = prompt.replace(protocol, "")
    assert rest.count("overall_notes") == 1 and "overall_notes" in BRIDGE
    # the record-field instructions never ask for an evidence field (the one use is the protocol's rule that a low
    # level of a construct is never missing evidence)
    fields = prompt[prompt.index("RECORD FIELDS"):prompt.index("CODEBOOK — PANEL INTERACTION")]
    assert "overall_notes" not in fields
    assert fields.count("evidence") == fields.count("never missing evidence") == 1


def test_the_stage_a_heading_names_the_delivery_and_states_the_scale_once():
    _, books, prompt = _books_and_prompt()
    heading = "CODEBOOK — STAGE A, DELIVERY OF THE PREPARED PRESENTATION\n"
    assert prompt.count(heading) == 1 and "PREPARED PRESENTATION ONLY" not in prompt
    section = prompt[prompt.index(heading):prompt.index("CODEBOOK — VENTURE CONTEXT")]
    assert section.count("whole number 1 to 5") == 1
    assert section.index("Each item: a whole number 1 to 5 from the anchors below.") < section.index(
        f"### {books.stage_a.entries[0].id}")
    # no Stage A definition speaks of "the video", so the line that scoped it is gone
    assert '"the video" means' not in prompt
    for e in books.stage_a.entries:
        assert "the video" not in " ".join((e.definition, e.include_if, e.exclude_if)).lower(), e.id
    fields = _flat(prompt[prompt.index("RECORD FIELDS"):prompt.index("CODEBOOK — PANEL INTERACTION")])
    assert "Stage A section" not in fields
    assert "protocol under STAGE A, DELIVERY OF THE PREPARED PRESENTATION below." in fields


def test_record_field_descriptions_follow_the_schema_order():
    from pipeline.schemas.stage_b_panel import PanelRecord
    _, _, prompt = _books_and_prompt()
    section = prompt[prompt.index("RECORD FIELDS"):prompt.index("CODEBOOK — PANEL INTERACTION")]
    order = ["actors", "prepared_spans", "excluded_spans", "gaps", "source_complete", "stage_a", "prepared", "venture",
             "turns", "exchanges", "decisions", "claim_conflicts", "notes"]
    assert list(PanelRecord.model_fields) == order
    positions = [section.index(f"\n{name} — ") for name in order]
    assert positions == sorted(positions)


def test_nameless_codes_render_the_id_once():
    _, books, prompt = _books_and_prompt()
    nameless = [e for e in books.stage_a.entries if not e.name]
    assert nameless
    for e in nameless:
        assert f"### {e.id}\n" in prompt
        assert f"{e.id} — {e.id}" not in prompt
    named = next(e for e in books.panel.entries if e.name)
    assert f"### {named.id} — {named.name}\n" in prompt


def test_choices_omit_none_where_the_record_may_not_use_it():
    _, books, prompt = _books_and_prompt()
    def choices(code_id):
        block = prompt[prompt.index(f"### {code_id} — "):]
        return next(ln for ln in block.splitlines() if ln.startswith("Answer with exactly one of: "))
    assert choices("EX-KIND") == "Answer with exactly one of: substantive | social | demonstration | negotiation"
    assert choices("DEC-KIND") == ("Answer with exactly one of: offer | joins_offer | revises_offer | withdraws_offer "
                                   "| exit | re_entry | founder_accepts | founder_rejects | founder_counters")
    for code_id in ("TURN-FUNC", "EX-TOPIC", "RS-RESP", "VC-INDUSTRY"):
        assert choices(code_id).endswith("| NONE"), code_id
        assert choices(code_id) == "Answer with exactly one of: " + " | ".join(
            next(b for b in (books.panel, books.venture) if code_id in b.ids).by_id(code_id).values)


def test_follow_up_wording():
    _, _, prompt = _books_and_prompt()
    flat = _flat(prompt)
    assert "not_applicable (the code does not apply to this turn)" in flat
    assert "this event" not in prompt[:prompt.index("CODEBOOK — PANEL INTERACTION")]
    assert ("give the later turns the earlier exchange's id. Follow-up questions that build on the previous answer "
            "stay in the same exchange, whoever asks them. A joke that breaks into the questioning is its own social "
            "exchange. There is one negotiation exchange for the whole encounter: every turn that states an offer, "
            "joins one, replies to one or discusses terms belongs to it. A question about the ask or the valuation "
            "before any offer is a question with deal_terms in a substantive exchange; the negotiation exchange "
            "begins with the first offer. An investor's statement that they are out, "
            "with the reasons they give, is a single decides turn that belongs to no exchange. decisions — ") in flat
    for phrase in ("build on the previous answer", "one negotiation exchange"):
        assert phrase in flat, phrase
    assert "never decides how that investor left" not in flat and "EX-CLOSE" not in prompt
    assert "unless it is spoken as part of the negotiation" not in flat
    assert "the terms as spoken, and a note holding the reasons the investor gives. Every offer gets" in flat
    assert "the terms as spoken, a note." not in flat


def test_v3_dev6_wording():
    _, _, prompt = _books_and_prompt()
    flat = _flat(prompt)
    assert ("covering the prepared presentation, the questioning, social talk, demonstrations and the negotiation. "
            "The prepared presentation is always recorded, as one or more presents turns. Omit bare listener "
            "tokens") in flat
    assert ('including capitals (NEUTRAL, NONE). Never give a value together with a missing reason other than '
            '"none". 7. Rate each turn') in flat


def test_prompt_asks_for_concepts_and_no_frame():
    _, _, prompt = _books_and_prompt()
    assert "PR-FRAME" not in prompt and "RS-FRAME" not in prompt
    ours = prompt[:prompt.index("CODEBOOK — PANEL INTERACTION")].lower()
    for word in ("promotion", "prevention", "framing", "regulatory"):
        assert word not in ours, word
    assert "Required on every `question` turn." in prompt[prompt.index("### PR-CONCEPT"):prompt.index("### PR-SEV")]
    assert "Required on every `answer` turn." in prompt[prompt.index("### RS-CONCEPT"):prompt.index("### RS-EVID")]


def test_shipped_codebooks_carry_only_the_real_question_concept_and_answer_directness_examples():
    _, books, prompt = _books_and_prompt()
    carrying = [e.id for book in (books.panel, books.stage_a, books.venture) for e in book.entries if e.examples]
    assert carrying == ["PR-CONCEPT", "RS-RESP"]
    assert [x.provenance for x in books.panel.by_id("PR-CONCEPT").examples] == ["real"] * 4
    assert [x.provenance for x in books.panel.by_id("RS-RESP").examples] == ["real"] * 2
    assert prompt.count("Examples:") == 2


def test_prompt_carries_the_two_real_answer_directness_examples_and_no_source():
    _, _, prompt = _books_and_prompt()
    block = prompt[prompt.index("### RS-RESP"):prompt.index("### RS-CONCEPT")]
    examples = block[block.index("Examples:"):].strip().splitlines()
    assert examples[0] == "Examples:" and len(examples) == 3
    assert examples[1].startswith("- ") and examples[1].endswith("→ partial")
    assert examples[2].startswith("- ") and examples[2].endswith("→ addresses")
    assert block.index("Exclude if:") < block.index("Examples:")
    for source in ("[real", "st_1164b596f8", "st_b76e8a6734", "turn 25", "turn 52"):
        assert source not in prompt, source


def test_prompt_carries_the_four_real_examples_and_no_source():
    _, _, prompt = _books_and_prompt()
    block = prompt[prompt.index("### PR-CONCEPT"):prompt.index("### PR-SEV")]
    assert block.rstrip().endswith(
        "Examples:\n"
        "- Kevin questions the high valuation, asking why he is asking for $1M for 5%. → deal_terms\n"
        "- Daymond asks if he is investing in a piece of the original design firm. → deal_terms\n"
        "- Peter asks why they made the corporate structure so complicated. → deal_terms\n"
        "- Kevin asks what the current customer acquisition cost is. → operating_efficiency (a cost, not acquisition)")
    for source in ("st_aebb7e3764", "st_8642720802", "turn 10", "turn 61", "turn 67", "turn 31", "[real"):
        assert source not in prompt, source


def test_only_real_examples_reach_the_model_and_never_their_source():
    from dataclasses import replace
    from pipeline.codebook import Example
    release, books, plain = _books_and_prompt()

    def with_examples(code_id, examples):
        entries = [e.model_copy(update={"examples": examples}) if e.id == code_id else e for e in books.panel.entries]
        return render_panel_prompt(replace(books, panel=books.panel.model_copy(update={"entries": entries})),
                                   release.templates_dir)

    real = Example(text="Asking what one unit costs to make is operating_efficiency.", provenance="real",
                   source="st_38f8bf18e0 turn 14")
    second = Example(text="Asking who the competitors are is competition.", provenance="real", source="video 7")
    made_up = Example(text="A made-up question about rocket fuel.", provenance="illustrative", source="invented")
    prompt = with_examples("PR-CONCEPT", [real, made_up, second])
    block = prompt[prompt.index("### PR-CONCEPT"):prompt.index("### PR-SEV")]
    assert block.rstrip().endswith("Examples:\n- Asking what one unit costs to make is operating_efficiency.\n"
                                   "- Asking who the competitors are is competition.")
    assert block.index("Exclude if:") < block.index("Examples:")
    for hidden in ("rocket fuel", "st_38f8bf18e0", "video 7", "invented", "illustrative"):
        assert hidden not in prompt, hidden
    assert prompt.count("Examples:") == 2  # the shipped RS-RESP block stays beside the swapped PR-CONCEPT one
    # nothing else moved: without the three added lines the prompt is the shipped one
    assert prompt.replace(block, plain[plain.index("### PR-CONCEPT"):plain.index("### PR-SEV")]) == plain
    only_made_up = with_examples("PR-CONCEPT", [made_up])
    assert only_made_up == with_examples("PR-CONCEPT", [])
    assert "Examples:" not in only_made_up[only_made_up.index("### PR-CONCEPT"):only_made_up.index("### PR-SEV")]


def test_v3_dev8_wording():
    _, books, prompt = _books_and_prompt()
    flat = _flat(prompt)
    assert ("(the category, the whole number, or true/false) and the missing reason. When a question or answer "
            "concept is other, name the matter in a few words in other_concept; otherwise other_concept is null. "
            "exchanges — ") in flat
    assert ("For each exchange give its kind; its topic from EX-TOPIC, which uses the same concepts as the question "
            "concept (NONE unless the exchange is substantive); a label of a few words naming the specific matter, "
            "which is required when the topic is other. Turns point to their exchange through exchange_id;") in flat
    for code_id in ("PR-CONCEPT", "RS-CONCEPT", "EX-TOPIC"):
        block = prompt[prompt.index(f"### {code_id} — "):]
        line = next(ln for ln in block.splitlines() if ln.startswith("Answer with exactly one of: "))
        assert line.endswith("| team_capabilities | deal_terms | other | NONE"), code_id


def test_gender_composition_code_renders_and_is_asked_for():
    _, books, prompt = _books_and_prompt()
    assert books.venture_values("VC-GENDER") == ("all_women", "all_men", "mixed", "NONE")
    block = prompt[prompt.index("### VC-GENDER — Presenting team gender composition"):]
    assert "Answer with exactly one of: all_women | all_men | mixed | NONE" in block
    assert "it is not a statement of anyone's gender identity" in _flat(block)
    assert "Project control variable" not in prompt and "Kanze" not in prompt
    assert ("venture — a short neutral description of what is sold and to whom; the primary industry, product form, "
            "customer type and the presenting team's gender composition from the codes below; any additional "
            "supported industries;") in _flat(prompt)


def test_prepared_presentation_codes_render_once_in_their_own_section():
    _, books, prompt = _books_and_prompt()
    assert [e.id for e in books.prepared_codes()] == ["PRES-SUBS", "PRES-COH", "PRES-CTX", "PRES-EVID"]
    assert books.codes_for("presents") == ()
    assert all(e.unit == "PITCH" and e.applies_to == "prepared" for e in books.prepared_codes())
    heading = "CODEBOOK — PREPARED PRESENTATION (judged from the prepared presentation spans only)"
    assert prompt.count(heading) == 1
    assert prompt.index("CODEBOOK — PANEL INTERACTION") < prompt.index(heading) < prompt.index("CODEBOOK — STAGE A")
    section = prompt[prompt.index(heading):prompt.index("CODEBOOK — STAGE A")]
    for e in books.prepared_codes():
        assert prompt.count(f"### {e.id} — ") == 1 and f"### {e.id} — " in section
    evid = section[section.index("### PRES-EVID"):]
    assert "Answer with a whole number, 0 or more." in evid
    assert "Recorded as a list" not in evid and "Required on every `prepared` turn" not in prompt
    consist = prompt[prompt.index("### PREP-CONSIST"):]
    assert consist.index("Recorded as a list; see the record fields.") < consist.index("Definition:")
    fields = prompt[prompt.index("RECORD FIELDS"):prompt.index("CODEBOOK — PANEL INTERACTION")]
    assert ("prepared — exactly one item for each code under PREPARED PRESENTATION below, judged from the prepared "
            "presentation spans only, before any investor speaks") in " ".join(fields.split())
    for marker in ("Chen", "Pollack", "Yao", "Kotha"):
        assert marker not in prompt


def _block(prompt, code_id):
    start = prompt.index(f"### {code_id} — ")
    return _flat(prompt[start:prompt.index("\n### ", start + 1)])


def test_deal_terms_boundaries():
    _, _, prompt = _books_and_prompt()
    for code_id in ("PR-CONCEPT", "RS-CONCEPT", "EX-TOPIC"):
        block = _block(prompt, code_id)
        assert "none of the concepts above covers" in block, code_id
        assert "none of the sixteen concepts" not in block, code_id
    assert ("Known close calls: the price of the product is sales, but the price of the company — the ask, the "
            "valuation, the equity — is deal_terms;") in _block(prompt, "PR-CONCEPT")
    assert ("Exclude if: Interest and conditional remarks without a stated offer (\"I'd do it if you had sales\") are "
            "not offers; record them as `accepts` turns. A compliment is a `social` turn. Valuation discussion before "
            "any offer is a `question` turn carrying `deal_terms`, or a `negotiates` turn once an offer stands.") in \
        _block(prompt, "DEC-KIND")
    assert "Interest, compliments, valuation discussion" not in _flat(prompt)
    rule = ("A question about the ask or the valuation before any offer is a question with deal_terms in a "
            "substantive exchange; the negotiation exchange begins with the first offer.")
    ex_kind = _block(prompt, "EX-KIND")
    assert rule in ex_kind[ex_kind.index("Include if:"):ex_kind.index("Exclude if:")]
    fields = _flat(prompt[prompt.index("RECORD FIELDS"):prompt.index("CODEBOOK — PANEL INTERACTION")])
    assert rule in fields
    assert ("is one decides turn and is never split into a challenge or an accepts turn plus a decision.") in \
        _block(prompt, "TURN-FUNC")


def test_the_ask_is_given_in_per_cent_and_a_royalty_keeps_its_amount():
    _, _, prompt = _books_and_prompt()
    block = _flat(prompt[prompt.index("### VC-ASK"):])
    assert ("Include if: Take the first complete statement of the ask. Give ask_equity_pct in per cent: 10 for ten "
            "per cent, never 0.10. A royalty or loan structure in the opening is recorded with the amount as stated "
            "and equity 0, with the structure in the venture note.") in block


def test_the_preamble_records_the_hash_of_the_rendered_prompt():
    """The revision line of the panel book's preamble names the prompt hash of this release. The preamble is
    documentation and never reaches the model, so recording the hash there does not change it."""
    import re
    release, books, prompt = _books_and_prompt()
    preamble = release.books["stage_b_panel"].markdown.read_text().split("\n## ", 1)[0]
    assert "Revision v3-dev9" not in prompt
    line = next(ln for ln in preamble.splitlines() if ln.startswith("Revision v3-dev9 (2026-10-20)"))
    assert re.findall(r"prompt hash ([0-9a-f]{16})\.", line) == [prompt_hash(prompt)]
