import pandas as pd

from pipeline.stage_b.examples import candidates_markdown, find_candidates


def ann(video, episode, k, code, value=None, label=None, refs="[2]", uncodable=False, valid=True):
    return {"pass": "episode", "video_id": video, "episode_idx": episode, "replicate_k": k,
            "code_id": code, "value": value, "value_label": label, "uncodable": uncodable,
            "evidence": "asks about margins", "turn_refs": refs, "turn_refs_valid": valid}


_TAGS = ["greets the panel, names the product", "asks 100k for ten percent",
         "asks about margins", "states sixty percent gross margin"]

TURNS = pd.DataFrame([
    {"video_id": v, "turn_idx": i, "speaker": s, "role": r, "start_s": float(i), "end_s": i + 1.0, "tag": t}
    for v in ("v1", "v2")
    for i, (s, r, t) in enumerate([("speaker-1", "founder", _TAGS[0]),
                                   ("speaker-1", "founder", _TAGS[1]),
                                   ("speaker-2", "investor", _TAGS[2]),
                                   ("speaker-1", "founder", _TAGS[3])])
])


def test_candidates_prefer_agreement_and_quote_the_cited_turns():
    annotations = pd.DataFrame([
        ann("v1", 0, 0, "PR-FRAME", label="prevention"), ann("v1", 0, 1, "PR-FRAME", label="prevention"),
        ann("v2", 0, 0, "PR-FRAME", label="prevention"), ann("v2", 0, 1, "PR-FRAME", label="promotion"),
        ann("v1", 0, 0, "RS-EVID", value=1.0, refs="[3]"),
        ann("v1", 0, 0, "PREP-SUBS", uncodable=True, refs="[]"),
        ann("v1", 0, 0, "RAP-X", value=2.0, refs="[9]", valid=False),
    ])
    found = find_candidates(annotations, TURNS)
    assert set(found.code_id) == {"PR-FRAME", "RS-EVID"}           # uncodable and invalid refs dropped
    frame = found[found.code_id == "PR-FRAME"]
    assert list(frame.video_id) == ["v1", "v2"] and list(frame.agreed) == [True, False]
    assert frame.iloc[0].excerpt == "[t2 0:02] speaker-2 (investor): asks about margins"
    assert found[found.code_id == "RS-EVID"].iloc[0].answer == "1"
    assert list(find_candidates(annotations, TURNS, dev_ids=["v2"]).video_id.unique()) == ["v2"]
    text = candidates_markdown(found)
    assert "## PR-FRAME" in text
    assert "- [real | v1 t2 0:02] TYPE THE VERBATIM EXCERPT HERE" in text
    assert "CHECK AGAINST THE VIDEO" in text
