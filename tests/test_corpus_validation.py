import json

from analysis import corpus_validation as validation, demo


def report():
    return json.loads((demo.REFERENCE / "review/agreement/validation_table.json").read_text())


def test_validation_table_preserves_statistics_scales_and_distinct_denominators():
    data = report()
    table = validation.agreement_table(data)
    assert len(table) == 12
    assert data["note"]["videos"] == 10
    assert table.Family.drop_duplicates().tolist() == [f for f, _ in validation.FAMILIES]
    for family, codes in validation.FAMILIES:
        for code in codes:
            row = table[table.Code.str.contains(f"({code})", regex=False)].iloc[0]
            for key, col in [("rows", "Gemini vs. human: AC (α), 95% CI"),
                             ("human_human_rows", "Human1 vs. human2: AC (α), 95% CI")]:
                source = next(r for r in data[key] if r["field"] == code)
                statistic = "ac2" if source["kind"] == "ordinal" else "ac1"
                assert f"{statistic.upper()} {source['figures'][statistic]['value']:.2f}" in row[col]
                assert f"(α {source['alpha']['value']:.2f})" in row[col]
    frame = table[table.Code.str.contains("PR-FRAME")].iloc[0]
    assert "unclassified" in frame.Scale and "NEUTRAL" in frame.Scale
    pitch = table[table.Code.str.contains("PRES-SUBS")].iloc[0]
    assert pitch["Gemini–human: units / judgments"] == "10 / 20"
    assert pitch["Human–human: units"] == 10
    html = validation.styled_agreement(table).to_html()
    assert "AV" not in html and "MX" not in html
    assert "AC2 0.75 (α -0.42)" in html


def test_additional_checks_keep_original_measures_without_fabricated_alpha():
    table = validation.additional_checks(report())
    assert len(table) == 5
    assert not table["Gemini vs. human"].str.contains("AC1|AC2").any()
    clock = table[table.Code.str.contains("First decision")].iloc[0]
    assert "8/10" in clock["Gemini vs. human"]
    assert "5/5" in clock["Human1 vs. human2"]
