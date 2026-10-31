"""Keep whole-record denominators distinct from episode counts and model features."""
from types import SimpleNamespace

import pandas as pd

from analysis import corpus_overview as corpus, demo, panel_explorer as px
from analysis.panel_bottleneck import LADDER, ladder_columns, stage_a_fallback


def test_episode_statistics_include_zero_events_and_exclude_missing_asks():
    data = SimpleNamespace(
        videos=pd.DataFrame({"video_id": ["a", "b"], "duration_s": [60, 120],
                             "ask_amount_usd": [100, None], "ask_equity_pct": [10, None]}),
        turns=pd.DataFrame({"video_id": ["a", "a", "b"], "function": ["question", "answer", "presents"]}),
        decisions=pd.DataFrame({"video_id": ["a", "a"]}),
    )
    summary = corpus.episode_summary(data).set_index("Measure")
    assert summary.loc["Questions per episode", "Mean"] == .5
    assert summary.loc["Decisions per episode", "Mean"] == 1
    assert summary.loc["Investment asked (USD; VC-ASK)", "Mean"] == 100
    assert summary.loc["Investment asked (USD; VC-ASK)", "Missing"] == 1


def test_full_corpus_code_counts_cover_codebook_and_match_saved_analysis():
    folder = demo.REFERENCE / "analysis/all100"
    features = pd.read_csv(folder / "features.csv")
    books = px.instrument(demo.REFERENCE / demo.EXAMPLE_ID).books
    counts = px.code_value_counts(demo.REFERENCE, books, features.video_id.tolist())
    expected_codes = {e.id for book in (books.panel, books.stage_a, books.venture) for e in book.entries}
    # VC-ASK has continuous dollar/equity summaries in episode_summary.
    assert set(counts.code) == expected_codes - {"VC-ASK"}
    grouped = counts.groupby("code")
    assert grouped["count"].sum().equals(grouped.n.first())
    assert grouped.n.first()["TURN-FUNC"] == len(pd.read_csv(folder / "turns.csv")) == 6730
    assert grouped.n.first()["DEC-KIND"] == len(pd.read_csv(folder / "decisions.csv")) == 746
    assert grouped.n.first()["PRES-SUBS"] == len(features) == 100
    assert counts.loc[(counts.code == "PREP-SUBS") & (counts.kind == "missing"), "count"].sum() > 0
    assert not counts.kind.eq("not in codebook").any()


def test_overview_families_match_figure2_and_omit_exchanges():
    features = pd.read_csv(demo.REFERENCE / "analysis/all100/features.csv")
    families = corpus.model_family_columns(features)
    models, _ = stage_a_fallback(LADDER, features)
    all_columns = set(ladder_columns(features, models["C5"]))
    assert set(c for columns in families.values() for c in columns) == {
        c for c in all_columns if not c.endswith("_missing")}
    assert list(families) == ["C2p", "C3q", "C3a", "C4r", "C5r"]
    assert "rx_receipt_share" in families["C4r"]
    assert "founder_repairs" in families["C5r"]
    assert not any(code.startswith("EX-") for _, _, codes in corpus.MODEL_GROUPS for code in codes)
    assert not any("exchange" in c for columns in families.values() for c in columns)


def test_family_statistics_do_not_count_missing_placeholders_as_zero():
    features = pd.DataFrame({"q_sev_trend": [0., 2.], "q_sev_trend_missing": [1, 0]})
    row = corpus.family_statistics(features, ["q_sev_trend"]).iloc[0]
    assert row["Mean"] == 2
    assert row["Missing"] == 1
    assert row["Episodes with data"] == 1
