# Pitch-level analysis of full-corpus-100

Written by `scripts/analysis/analyze_panel.py` from the tables beside this file; every number here is in one of them.

## 0. The pre-registered test

Quoted verbatim from `docs/plans/design docs/2026-10-19-pitch-level-refocus.md`, section 6 (the file's sha256 afaedc5c06387d05ddd569f395674f2e240242ad971f79600f383dd252826d48). In the code C2′ is `C2p`.

Written before the v3-dev9 annotations are analysed; the script prints the verdict from the rule. Amended on
2026-10-21 (6.8, at the end of this section).

**Rows.** 100 pitches, `any_offer` (72 with an offer decision in the current run, 28 without; the reaction-test report counts 27 under its `ever_offer`, which includes joined offers; the count on v3-dev9 may differ by a
few after re-annotation and is reported). Leave-one-pitch-out; inner-fold penalty; 1,999 bootstrap draws; 500
permutations; seed 42.

**Ladder.**

| Model | Adds | Features |
|---|---|---|
| C0 | base rate | — |
| C1 | venture + the ask + exposure | industry, product form, customer type, team gender; ask amount (log) and equity; turns, seconds, questions before the cut |
| C2 | + prepared delivery | the `pitch_positivity` rank score (enthusiasm and warmth as a sensitivity; decision 9) |
| C2′ | + prepared content | the four `PREP-PRES` ratings of the prepared presentation |
| C3q | + questions only | questions, questioners, mean severity, share hard, concept-frame shares (promotion, prevention, deal terms, other) |
| C3 | + answers | directness mean, evidence share, uncertainty share, substance mean, coherence mean, context mean; substance and evidence on hard questions |
| C4 | + reactions | per-kind reaction shares (`rx_all_accepts`, `rx_all_follow_up`, `rx_all_challenge`) and the challenge and receipt next-move shares |
| C5 | + rapport and repair | social turns, hostile turns, repair turns |

**Comparisons (later minus earlier), pooled AUC and log-loss skill, paired pitch bootstrap:** C2 − C1, C2′ − C2,
**C3 − C2′ (H1′, primary)**, **C3 − C3q (H8, primary)**, C4 − C3 (H2′), C5 − C4.

**Decision rule.**
- H1′ supported when C3 − C2′ has an AUC gain of at least +0.05 with an interval excluding 0, or a skill gain whose
  interval excludes 0.
- H8 supported when C3 − C3q has an AUC gain of at least +0.03 with an interval excluding 0.
- Each is reported as *not distinguishable from zero* otherwise; with 28 no-offer pitches the power statement
  (section 6.5) is printed beside every verdict.

**6.4 Sensitivities, each a row beside the primary.** (a) Both passes averaged for every numeric feature (the
diagnosis's "noise halved for free"); (b) exposure as rates, not counts; (c) the 95 non-development videos only;
(d) fixed penalty 10; (e) once human validation is imported, the same ladder on reviewed records.

**6.5 Power.** From the v3-dev8 intervals, a pooled-AUC difference of about ±0.07 is the resolution at 100 pitches:
the rule's +0.05 and +0.03 are what the data can see, not what the theory predicts. The report states this.

**6.7 H9, the answer-level test (pre-registered with the ladder).** Unit: an answer to an investor question before
the pitch's cut (about 1,500; 1,542 answers to investor questions on `v3-dev8`). Predictors: the answer's substance, checkable support, coherence, context, and
admitted uncertainty; the question's severity as a moderator. Outcomes, each read from turn order inside the
exchange and the pitch: (i) the next investor turn is a challenge, an accept, or a further question (H4's panel
next move, now with the merged `accepts`); (ii) the next question's severity; (iii) whether the next question stays
in the exchange (a follow-up) or opens a new one; (iv) the number of further questions in the exchange. Models:
pitch-stratified (conditional) logit for (i) and (iii), pitch-fixed-effect ordinal or linear for (ii) and (iv),
intervals from resampling pitches; reported for hard questions (severity 2–3) and all questions. The responsiveness
test: whether `RS-UNCERT` concessions and `repair` turns are followed by more or fewer further questions and by a
challenge or an accept, the behavioural form of Pollack et al.'s negative responsiveness coefficient. No decision
rule: H9 is reported with intervals, and a direction is claimed only when the interval excludes zero for both hard
and all questions. The pitch-level path (answer preparedness → questioning effort → offer) is described, not tested
as a mediation: 100 pitches cannot carry an indirect-effect test, and the report says so.

**6.6 Outputs.** `runs/<v3-dev9 run>/analysis/pitch_level/`: metrics, deltas, predictions, permutations,
coefficients, single features, H4 and H5 tables, figures, `report.md` with this section verbatim and the verdicts.

**6.8 Amendment 2026-10-21 (the researcher's decisions).** Written before the ladder or H9 is run on `v3-dev9`, but
after the `v3-dev9` annotations exist and after one look at H4 under both readings of the next move (in the exchange,
on turn order). Where this block and the text above differ, this block holds; the text above is kept as written. The
reasons are in `docs/plans/project_plan_v4.md`, section 7, first paragraph: the model's exchange grouping is unstable
between passes (co-membership median 0.75, minimum 0.32) and groups the same line of questions differently in
different pitches.
1. **Reactions and the next move on turn order.** C4's reaction and next-move shares, H4 and H9 read turn order, not
   the exchange. The next move is the first investor turn after the answer (social and `NONE` turns and founder
   continuations passed over); a `decides` or `negotiates` turn there, or no investor turn at all, gives
   `no_reaction`, and a next turn at or after the cut gives `after_cut`. A reaction window runs from the answer to the
   next founder answer or to decision talk. A reaction that returns to an earlier answer is credited to the nearest
   preceding answer (plan v4, section 11).
2. **Close and distant questions are the frame only.** No topics are derived and no other distance between concepts is
   defined: the frame (promotion or prevention) is the only grouping of concepts, and entropy over concepts the only
   diversity measure. Decision 2 of section 9 is overtaken.
3. **Sequence features in C3q; H9 on the frame.**
   - Questions family (all `o1`, declared with `_declare_stat`, fitted by C3q and every later model):
     `q_sev_trend` (mean severity of the later half of the investor questions in turn order minus the earlier half,
     via `_trend`; 0.0 with fewer than two), `q_pursuit_share` (share of consecutive investor-question pairs, in turn
     order, with the same asker and a strictly higher severity on the second; NaN with no pair),
     `q_frame_switch_share` (share of consecutive investor-question pairs where both frames are promotion/prevention
     and they differ; NaN with no such pair), `q_concept_entropy` (Shannon entropy in bits of the concept
     distribution over investor questions whose concept is not `NONE`; 0.0 with one question; NaN with none).
     Exposure: `demo_seconds` now sums `demonstrates` turns; `n_social_exchanges` removed (C5's `social_turns`
     already counts social turns).
   - H9: `BINARY_OUTCOMES = ("next_accept", "next_challenge", "next_further_question", "next_question_same_frame")`,
     `CONTINUOUS_OUTCOMES = ("next_question_sev", "further_questions_same_frame")`,
     `SENSITIVITY_OUTCOMES = ("next_question_same_exchange", "further_questions_in_exchange")`. Outcome (iii) is
     whether the next investor question keeps the answered question's frame (promotion or prevention) or switches;
     (iv) is the number of consecutive later questions that keep that frame. `answer_table` gains `q_promotion` (0/1
     from the answered question's frame; NaN when unclassified) and the three new follow-up columns
     (`next_question_same_frame`, `further_questions_same_frame`, `next_question_same_concept`). For the two
     same-frame outcomes every nested model also enters `q_promotion` as a 0/1 covariate (it is not a predictor of
     interest and `h9_summary` leaves it out of directions). `h9_sensitivity` runs `SENSITIVITY_OUTCOMES` with the
     same models (`--h9-exchange-outcomes`, 499 draws by default, `h9_exchange_sensitivity.csv`): computed and not
     reported.
   - Hypothesis wording of H9: after a more prepared answer, the panel accepts or switches frame rather than
     challenging or pressing on in the same frame.

Also decided on 2026-10-21: no pre-registration ceremony, but every change is written into plan v4 with its date; the
exchange-based results are computed as sensitivities and not reported. `PRES-CTX` (C2′) and the answers' context
rating `PREP-CTX` (C3, H9's `full` model) are not human-coded and are labelled "Gemini only, not human-coded". H9 is
reported as an exploratory pattern, not a set of confirmatory findings.

**6.9 Amendment 2026-10-25 (the researcher's decision): every family against the baseline.** Written after the
`v3-dev9` run of 2026-10-21 and its results had been read, so what it adds is reported, not tested; it adds no
decision rule and changes neither H1′ nor H8. The reason: the proposal's question compares each family of concepts
with what the pitch shows, but the cumulative ladder compares each family only with the step before it, so the
reactions (C4 − C3) and rapport and repair (C5 − C4) were compared with the answers, not with the baseline. Three
one-family models are added to the ladder, each the baseline C2′ plus one family alone: **C3a** (+ answers only),
**C4r** (+ reactions only), **C5r** (+ rapport and repair only); C3q is already C2′ + questions only. The
comparisons add, later minus earlier, with the same paired pitch bootstrap: C3a − C2′, C4r − C2′, C5r − C2′, and
the cumulative C4 − C2′ and C5 − C2′ (C3q − C2′ and C3 − C2′ were already paired). The report gains a table "Gains
over the baseline C2′" (each family alone and cumulative, AUC and skill, with intervals and bootstrap p) and a
Figure 10 that draws it; every gain the proposal quotes is a gain over the baseline with its p-value. The
one-family models run through the same leave-one-pitch-out, permutation test and sensitivities as the rest of the
ladder; the ROC and calibration figures keep the cumulative models only. H2′ (reactions beyond content) is read as
C4r − C2′ and C4 − C2′ beside C4 − C3.

## 1. Sample and outcomes

- 100 pitches (model-coded: 100 pitches; human-reviewed: 0 pitches); label: pitch-held-out estimate.
- Outcome `any_offer` (any investor made or joined an offer): 73 with an offer, 27 without, 0 unknown.
- Unit: one row per pitch, cut at the first decision of any kind by any investor; leave-one-pitch-out, so every pitch is one independent observation.
- Stage A ranking (the rank score C2 reads): complete.

## 2. The ladder

Stage A: C2 reads the rank score of `pitch_positivity`, which every pitch has (complete).

Penalty: inner_cv; 1999 bootstrap draws; 500 permutations (permutation p: does the model beat the offer rate, in log loss and in AUC).

AUCs (and their intervals, differences and permutation p-values) are computed on the base-rate-adjusted held-out score logit(p) − logit(training base rate of the pitch's fold), so they are not comparable with the AUCs of earlier (v3-dev8) analyses, which ranked the raw probabilities; log loss, Brier score and skill read the probabilities.

### Metrics

| Model | Families | Skill % | Skill 95% | AUC | AUC 95% | AUC meaningful | Log loss | Penalty | Permutation p (log loss) | Permutation p (AUC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | (base rate) | 0.0 | [0.0, 0.0] | 0.500 | [0.500, 0.500] | no | 0.593 | none fitted | — | — |
| C1 | control+exposure | -6.4 | [-10.7, -2.3] | 0.424 | [0.287, 0.564] | yes | 0.631 | inner_cv: 10 x38, 100 x36, 1000 x26 | 0.385 | 0.369 |
| C2 | control+exposure+sa_rank_pitch_positivity | -6.7 | [-10.9, -2.8] | 0.434 | [0.305, 0.576] | yes | 0.633 | inner_cv: 10 x18, 100 x38, 1000 x44 | 0.555 | 0.503 |
| C2p | control+exposure+sa_rank_pitch_positivity+prepared | -1.6 | [-4.1, 0.8] | 0.441 | [0.295, 0.588] | no | 0.603 | inner_cv: 10 x1, 100 x78, 1000 x21 | 0.333 | 0.543 |
| C3q | control+exposure+sa_rank_pitch_positivity+prepared+questions | -2.7 | [-9.3, 3.3] | 0.538 | [0.404, 0.667] | yes | 0.609 | inner_cv: 10 x11, 100 x89 | 0.064 | 0.200 |
| C3 | control+exposure+sa_rank_pitch_positivity+prepared+questions+answers | 8.8 | [-5.3, 22.3] | 0.685 | [0.563, 0.798] | yes | 0.541 | inner_cv: 10 x100 | 0.004 | 0.004 |
| C4 | control+exposure+sa_rank_pitch_positivity+prepared+questions+answers+reactions | 7.1 | [-7.7, 20.9] | 0.701 | [0.583, 0.801] | yes | 0.551 | inner_cv: 10 x93, 100 x7 | 0.006 | 0.006 |
| C5 | control+exposure+sa_rank_pitch_positivity+prepared+questions+answers+reactions+rapport_repair | 4.2 | [-13.5, 20.7] | 0.685 | [0.553, 0.801] | yes | 0.569 | inner_cv: 10 x93, 100 x7 | 0.008 | 0.006 |
| C3a | control+exposure+sa_rank_pitch_positivity+prepared+answers | 6.9 | [-4.8, 18.2] | 0.666 | [0.551, 0.776] | yes | 0.553 | inner_cv: 10 x98, 100 x2 | 0.006 | 0.014 |
| C4r | control+exposure+sa_rank_pitch_positivity+prepared+reactions | 0.3 | [-12.3, 11.6] | 0.592 | [0.456, 0.725] | yes | 0.592 | inner_cv: 10 x84, 100 x16 | 0.028 | 0.052 |
| C5r | control+exposure+sa_rank_pitch_positivity+prepared+rapport_repair | 0.5 | [-9.7, 10.4] | 0.568 | [0.430, 0.705] | yes | 0.590 | inner_cv: 10 x99, 100 x1 | 0.068 | 0.178 |

"AUC meaningful" is a diagnostic only (ruling I3 of plan B's final review): "no" marks a model whose held-out predictions stay within about 3 points of the offer rate; it withholds nothing, and that model's AUC counts in the verdicts. The AUC route is withheld only when a model's AUC interval lies wholly below 0.5.

### Paired differences (later minus earlier)

| Comparison | Metric | Difference | 95% interval | Bootstrap p |
| --- | --- | --- | --- | --- |
| C2 minus C1 | log_loss | +0.002 | [-0.006, +0.009] | 0.643 |
| C2 minus C1 | brier | +0.000 | [-0.003, +0.003] | 0.841 |
| C2 minus C1 | auc | +0.010 | [-0.056, +0.075] | 0.732 |
| C2 minus C1 | skill | -0.305 | [-1.461, +0.975] | 0.643 |
| C2p minus C2 | log_loss | -0.030 | [-0.051, -0.011] | 0.001 |
| C2p minus C2 | brier | -0.010 | [-0.017, -0.004] | 0.001 |
| C2p minus C2 | auc | +0.007 | [-0.090, +0.108] | 0.917 |
| C2p minus C2 | skill | +5.062 | [+1.878, +8.344] | 0.001 |
| C3 minus C2p | log_loss | -0.062 | [-0.144, +0.022] | 0.142 |
| C3 minus C2p | brier | -0.023 | [-0.056, +0.008] | 0.132 |
| C3 minus C2p | auc | +0.245 | [+0.081, +0.402] | 0.006 |
| C3 minus C2p | skill | +10.379 | [-3.811, +23.696] | 0.142 |
| C3 minus C3q | log_loss | -0.068 | [-0.137, +0.005] | 0.065 |
| C3 minus C3q | brier | -0.025 | [-0.052, +0.002] | 0.075 |
| C3 minus C3q | auc | +0.147 | [+0.032, +0.265] | 0.010 |
| C3 minus C3q | skill | +11.448 | [-0.869, +22.679] | 0.065 |
| C4 minus C3 | log_loss | +0.010 | [-0.036, +0.059] | 0.679 |
| C4 minus C3 | brier | +0.006 | [-0.012, +0.026] | 0.509 |
| C4 minus C3 | auc | +0.015 | [-0.049, +0.082] | 0.681 |
| C4 minus C3 | skill | -1.635 | [-9.818, +6.060] | 0.679 |
| C5 minus C4 | log_loss | +0.018 | [-0.030, +0.073] | 0.491 |
| C5 minus C4 | brier | -0.001 | [-0.019, +0.018] | 0.936 |
| C5 minus C4 | auc | -0.016 | [-0.081, +0.048] | 0.621 |
| C5 minus C4 | skill | -2.973 | [-12.031, +5.170] | 0.491 |
| C3q minus C2p | log_loss | +0.006 | [-0.028, +0.045] | 0.761 |
| C3q minus C2p | brier | +0.001 | [-0.011, +0.015] | 0.826 |
| C3q minus C2p | auc | +0.097 | [-0.054, +0.245] | 0.192 |
| C3q minus C2p | skill | -1.070 | [-7.444, +4.711] | 0.761 |
| C3a minus C2p | log_loss | -0.050 | [-0.113, +0.017] | 0.131 |
| C3a minus C2p | brier | -0.019 | [-0.045, +0.007] | 0.152 |
| C3a minus C2p | auc | +0.225 | [+0.090, +0.355] | 0.002 |
| C3a minus C2p | skill | +8.482 | [-2.779, +18.793] | 0.131 |
| C4r minus C2p | log_loss | -0.011 | [-0.078, +0.060] | 0.770 |
| C4r minus C2p | brier | -0.007 | [-0.033, +0.021] | 0.648 |
| C4r minus C2p | auc | +0.151 | [-0.002, +0.290] | 0.052 |
| C4r minus C2p | skill | +1.887 | [-10.368, +13.076] | 0.770 |
| C5r minus C2p | log_loss | -0.013 | [-0.070, +0.047] | 0.644 |
| C5r minus C2p | brier | -0.007 | [-0.030, +0.018] | 0.527 |
| C5r minus C2p | auc | +0.127 | [-0.011, +0.270] | 0.070 |
| C5r minus C2p | skill | +2.141 | [-7.893, +11.520] | 0.644 |
| C4 minus C2p | log_loss | -0.052 | [-0.134, +0.040] | 0.224 |
| C4 minus C2p | brier | -0.017 | [-0.049, +0.016] | 0.296 |
| C4 minus C2p | auc | +0.260 | [+0.099, +0.410] | 0.001 |
| C4 minus C2p | skill | +8.744 | [-6.316, +22.321] | 0.224 |
| C5 minus C2p | log_loss | -0.034 | [-0.130, +0.073] | 0.519 |
| C5 minus C2p | brier | -0.018 | [-0.054, +0.019] | 0.343 |
| C5 minus C2p | auc | +0.244 | [+0.074, +0.407] | 0.005 |
| C5 minus C2p | skill | +5.771 | [-11.827, +22.107] | 0.519 |

### Gains over the baseline C2p

Each family's gain over the baseline C2p (the venture, the ask, exposure, delivery and the prepared presentation), later minus earlier, with its 95% interval and bootstrap p: once when the family enters alone (the one-family model, amendment 6.9 of the spec) and once when it enters the cumulative ladder, on top of the families before it. Both are paired differences of the table above; the single steps of the cumulative ladder are there too.

| Family | Metric | Alone | Gain over C2p | Bootstrap p | Cumulative | Gain over C2p (cumulative) | Bootstrap p (cumulative) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| questions | auc | C3q | +0.097 [-0.054, +0.245] | 0.192 | C3q | +0.097 [-0.054, +0.245] | 0.192 |
| questions | skill | C3q | -1.070 [-7.444, +4.711] | 0.761 | C3q | -1.070 [-7.444, +4.711] | 0.761 |
| answers | auc | C3a | +0.225 [+0.090, +0.355] | 0.002 | C3 | +0.245 [+0.081, +0.402] | 0.006 |
| answers | skill | C3a | +8.482 [-2.779, +18.793] | 0.131 | C3 | +10.379 [-3.811, +23.696] | 0.142 |
| reactions | auc | C4r | +0.151 [-0.002, +0.290] | 0.052 | C4 | +0.260 [+0.099, +0.410] | 0.001 |
| reactions | skill | C4r | +1.887 [-10.368, +13.076] | 0.770 | C4 | +8.744 [-6.316, +22.321] | 0.224 |
| rapport_repair | auc | C5r | +0.127 [-0.011, +0.270] | 0.070 | C5 | +0.244 [+0.074, +0.407] | 0.005 |
| rapport_repair | skill | C5r | +2.141 [-7.893, +11.520] | 0.644 | C5 | +5.771 [-11.827, +22.107] | 0.519 |

### Design columns

The columns each model fits, on the pitches with a known outcome; a column constant in a training fold is dropped in that fold. Not fitted by any ladder model (descriptive columns of features.csv): every `_n` count behind a mean or share, the H5 frame-match share (a_frame_match_share), the exposure columns beyond the spec's C1 list (demo_seconds, prepared_seconds) and the count of answers with a reaction window (rx_answers), a denominator of the reaction shares. The context ratings (PRES-CTX in C2p, `pres_ctx`; PREP-CTX in C3 and H9's full model, `a_ctx_mean`, `ctx`) are Gemini only, not human-coded: they are not on the human coders' sheet.

| Model | Columns | Design columns |
| --- | --- | --- |
| C0 | 0 | (base rate) |
| C1 | 10 | founder_gender, primary_industry, product_form, customer_type, ask_log_amount, ask_equity_pct, ask_missing, hist_turns, hist_seconds, hist_questions |
| C2 | 11 | founder_gender, primary_industry, product_form, customer_type, ask_log_amount, ask_equity_pct, ask_missing, hist_turns, hist_seconds, hist_questions, sa_rank_pitch_positivity |
| C2p | 19 | founder_gender, primary_industry, product_form, customer_type, ask_log_amount, ask_equity_pct, ask_missing, hist_turns, hist_seconds, hist_questions, sa_rank_pitch_positivity, pres_subs, pres_subs_missing, pres_coh, pres_coh_missing, pres_ctx (Gemini only, not human-coded), pres_ctx_missing, pres_evid, pres_evid_missing |
| C3q | 40 | founder_gender, primary_industry, product_form, customer_type, ask_log_amount, ask_equity_pct, ask_missing, hist_turns, hist_seconds, hist_questions, sa_rank_pitch_positivity, pres_subs, pres_subs_missing, pres_coh, pres_coh_missing, pres_ctx (Gemini only, not human-coded), pres_ctx_missing, pres_evid, pres_evid_missing, q_askers, q_sev_mean, q_sev_mean_missing, q_hard_share, q_hard_share_missing, q_frame_promotion_share, q_frame_promotion_share_missing, q_frame_prevention_share, q_frame_prevention_share_missing, q_deal_terms_share, q_deal_terms_share_missing, q_other_share, q_other_share_missing, q_sev_trend, q_sev_trend_missing, q_pursuit_share, q_pursuit_share_missing, q_frame_switch_share, q_frame_switch_share_missing, q_concept_entropy, q_concept_entropy_missing |
| C3 | 56 | founder_gender, primary_industry, product_form, customer_type, ask_log_amount, ask_equity_pct, ask_missing, hist_turns, hist_seconds, hist_questions, sa_rank_pitch_positivity, pres_subs, pres_subs_missing, pres_coh, pres_coh_missing, pres_ctx (Gemini only, not human-coded), pres_ctx_missing, pres_evid, pres_evid_missing, q_askers, q_sev_mean, q_sev_mean_missing, q_hard_share, q_hard_share_missing, q_frame_promotion_share, q_frame_promotion_share_missing, q_frame_prevention_share, q_frame_prevention_share_missing, q_deal_terms_share, q_deal_terms_share_missing, q_other_share, q_other_share_missing, q_sev_trend, q_sev_trend_missing, q_pursuit_share, q_pursuit_share_missing, q_frame_switch_share, q_frame_switch_share_missing, q_concept_entropy, q_concept_entropy_missing, a_resp_mean, a_resp_mean_missing, a_evid_share, a_evid_share_missing, a_uncert_share, a_uncert_share_missing, a_subs_mean, a_subs_mean_missing, a_coh_mean, a_coh_mean_missing, a_ctx_mean (Gemini only, not human-coded), a_ctx_mean_missing, a_subs_hard_mean, a_subs_hard_mean_missing, a_evid_hard_share, a_evid_hard_share_missing |
| C4 | 72 | founder_gender, primary_industry, product_form, customer_type, ask_log_amount, ask_equity_pct, ask_missing, hist_turns, hist_seconds, hist_questions, sa_rank_pitch_positivity, pres_subs, pres_subs_missing, pres_coh, pres_coh_missing, pres_ctx (Gemini only, not human-coded), pres_ctx_missing, pres_evid, pres_evid_missing, q_askers, q_sev_mean, q_sev_mean_missing, q_hard_share, q_hard_share_missing, q_frame_promotion_share, q_frame_promotion_share_missing, q_frame_prevention_share, q_frame_prevention_share_missing, q_deal_terms_share, q_deal_terms_share_missing, q_other_share, q_other_share_missing, q_sev_trend, q_sev_trend_missing, q_pursuit_share, q_pursuit_share_missing, q_frame_switch_share, q_frame_switch_share_missing, q_concept_entropy, q_concept_entropy_missing, a_resp_mean, a_resp_mean_missing, a_evid_share, a_evid_share_missing, a_uncert_share, a_uncert_share_missing, a_subs_mean, a_subs_mean_missing, a_coh_mean, a_coh_mean_missing, a_ctx_mean (Gemini only, not human-coded), a_ctx_mean_missing, a_subs_hard_mean, a_subs_hard_mean_missing, a_evid_hard_share, a_evid_hard_share_missing, rx_reaction_rate, rx_reaction_rate_missing, rx_receipt_share, rx_receipt_share_missing, rx_followup_share, rx_followup_share_missing, rx_challenge_share, rx_challenge_share_missing, nm_receipt_share, nm_receipt_share_missing, nm_followup_share, nm_followup_share_missing, nm_challenge_share, nm_challenge_share_missing, accepts_turns, challenge_turns |
| C5 | 76 | founder_gender, primary_industry, product_form, customer_type, ask_log_amount, ask_equity_pct, ask_missing, hist_turns, hist_seconds, hist_questions, sa_rank_pitch_positivity, pres_subs, pres_subs_missing, pres_coh, pres_coh_missing, pres_ctx (Gemini only, not human-coded), pres_ctx_missing, pres_evid, pres_evid_missing, q_askers, q_sev_mean, q_sev_mean_missing, q_hard_share, q_hard_share_missing, q_frame_promotion_share, q_frame_promotion_share_missing, q_frame_prevention_share, q_frame_prevention_share_missing, q_deal_terms_share, q_deal_terms_share_missing, q_other_share, q_other_share_missing, q_sev_trend, q_sev_trend_missing, q_pursuit_share, q_pursuit_share_missing, q_frame_switch_share, q_frame_switch_share_missing, q_concept_entropy, q_concept_entropy_missing, a_resp_mean, a_resp_mean_missing, a_evid_share, a_evid_share_missing, a_uncert_share, a_uncert_share_missing, a_subs_mean, a_subs_mean_missing, a_coh_mean, a_coh_mean_missing, a_ctx_mean (Gemini only, not human-coded), a_ctx_mean_missing, a_subs_hard_mean, a_subs_hard_mean_missing, a_evid_hard_share, a_evid_hard_share_missing, rx_reaction_rate, rx_reaction_rate_missing, rx_receipt_share, rx_receipt_share_missing, rx_followup_share, rx_followup_share_missing, rx_challenge_share, rx_challenge_share_missing, nm_receipt_share, nm_receipt_share_missing, nm_followup_share, nm_followup_share_missing, nm_challenge_share, nm_challenge_share_missing, accepts_turns, challenge_turns, social_turns, founder_social, hostile_at_founder, founder_repairs |
| C3a | 35 | founder_gender, primary_industry, product_form, customer_type, ask_log_amount, ask_equity_pct, ask_missing, hist_turns, hist_seconds, hist_questions, sa_rank_pitch_positivity, pres_subs, pres_subs_missing, pres_coh, pres_coh_missing, pres_ctx (Gemini only, not human-coded), pres_ctx_missing, pres_evid, pres_evid_missing, a_resp_mean, a_resp_mean_missing, a_evid_share, a_evid_share_missing, a_uncert_share, a_uncert_share_missing, a_subs_mean, a_subs_mean_missing, a_coh_mean, a_coh_mean_missing, a_ctx_mean (Gemini only, not human-coded), a_ctx_mean_missing, a_subs_hard_mean, a_subs_hard_mean_missing, a_evid_hard_share, a_evid_hard_share_missing |
| C4r | 35 | founder_gender, primary_industry, product_form, customer_type, ask_log_amount, ask_equity_pct, ask_missing, hist_turns, hist_seconds, hist_questions, sa_rank_pitch_positivity, pres_subs, pres_subs_missing, pres_coh, pres_coh_missing, pres_ctx (Gemini only, not human-coded), pres_ctx_missing, pres_evid, pres_evid_missing, rx_reaction_rate, rx_reaction_rate_missing, rx_receipt_share, rx_receipt_share_missing, rx_followup_share, rx_followup_share_missing, rx_challenge_share, rx_challenge_share_missing, nm_receipt_share, nm_receipt_share_missing, nm_followup_share, nm_followup_share_missing, nm_challenge_share, nm_challenge_share_missing, accepts_turns, challenge_turns |
| C5r | 23 | founder_gender, primary_industry, product_form, customer_type, ask_log_amount, ask_equity_pct, ask_missing, hist_turns, hist_seconds, hist_questions, sa_rank_pitch_positivity, pres_subs, pres_subs_missing, pres_coh, pres_coh_missing, pres_ctx (Gemini only, not human-coded), pres_ctx_missing, pres_evid, pres_evid_missing, social_turns, founder_social, hostile_at_founder, founder_repairs |

## 3. Verdicts

### H1': C3 minus C2p

**Verdict: supported.** AUC difference +0.245 [+0.081, +0.402]; skill difference +10.38 [-3.81, +23.70] points. AUC route: used.

Passes on the AUC route only (the skill interval includes zero). The earlier models sit below chance under leave-one-pitch-out (C1 0.424, C2 0.434, C2p 0.441), the residual bias against weak models (section 8), so the AUC gain overstates the step: measured from 0.5, C3's AUC of 0.685 is +0.185 above chance.

With 27 pitches without an offer among 100, AUC differences smaller than about ±0.12 cannot be told from zero; the rule's thresholds (+0.05, +0.03) are below that resolution, so a pass rests on the interval excluding zero.

### H8: C3 minus C3q

**Verdict: supported.** AUC difference +0.147 [+0.032, +0.265]; skill difference +11.45 [-0.87, +22.68] points. AUC route: used.

Passes on the AUC route only (the skill interval includes zero; the rule has no skill route for this hypothesis).

With 27 pitches without an offer among 100, AUC differences smaller than about ±0.12 cannot be told from zero; the rule's thresholds (+0.05, +0.03) are below that resolution, so a pass rests on the interval excluding zero.

## 4. Sensitivities

Each row is a full evaluation of the ladder with the same draws and seed; the verdicts apply the same rules. The pre-registered verdicts are those of `main`.

| Setting | What changes | Pitches | H1' C3 minus C2p: AUC | H1': skill | H1' verdict | H8 C3 minus C3q: AUC | H8: skill | H8 verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | the pre-registered analysis | 100 | +0.245 [+0.081, +0.402] | +10.38 [-3.81, +23.70] | supported | +0.147 [+0.032, +0.265] | +11.45 [-0.87, +22.68] | supported |
| both_passes | both annotation passes averaged for every numeric feature (spec 6.4 a) | 100 | +0.203 [+0.048, +0.352] | +2.85 [-13.64, +15.10] | supported | +0.125 [+0.036, +0.217] | +2.74 [-11.23, +12.30] | supported |
| rates | exposure as rates per minute, not counts (spec 6.4 b) | 100 | +0.162 [+0.003, +0.328] | +9.98 [-4.50, +23.86] | supported | +0.131 [+0.014, +0.255] | +12.34 [-1.26, +25.42] | supported |
| non_development | without the pilot development videos named by the selection file (`--selection`, else the run's own; spec 6.4 c); every video was looked at while the instrument was shaped (plan v4 section 11), so this removes the pilot videos, not the looking | 95 | +0.272 [+0.113, +0.428] | +9.20 [-4.22, +22.81] | supported | +0.131 [+0.012, +0.254] | +6.89 [-5.24, +19.71] | supported |
| fixed_penalty_10 | ridge penalty fixed at 10 in every fold, not chosen inside it (spec 6.4 d) | 100 | +0.214 [+0.058, +0.374] | +12.94 [-0.45, +25.95] | supported | +0.135 [+0.020, +0.256] | +12.10 [+0.82, +23.68] | supported |
| all_stage_a | C2 adds the rank score of every Stage A item, not the primary one only (spec decision 9) | 100 | +0.250 [+0.092, +0.403] | +12.02 [-1.39, +24.67] | supported | +0.105 [-0.008, +0.224] | +5.58 [-6.52, +16.93] | not distinguishable from zero |

Every video has both passes for `both_passes`.

`non_development` leaves out the pilot videos the codebooks were first written on, named by the selection file (spec 6.4 c). In the wider sense of plan v4 section 11 every analysed video is a development video, since the instrument was shaped while looking at all of them; this sensitivity removes the pilot videos, not the looking, and does not make the analysis confirmatory.

## 5. H9

Coefficients compare answers within a pitch (conditional logit for the binary next moves, pitch fixed effects for the counts and the severity); per SD of a rating, per 0 to 1 of a binary predictor. No decision rule: a direction is claimed only where the intervals for all and for hard questions both exclude zero on the same side. The outcomes are read on turn order. Every model of next_question_same_frame and further_questions_same_frame also enters q_promotion (the answered question's frame, 1 promotion, 0 prevention), a covariate with no direction; answers to a question with neither frame are not in those models. Where the covariate separates the outcome or cannot be estimated in a population, the model is fitted without it and the Direction column says why. A predictor whose log-odds on the data lies beyond the bound the bootstrap draws obey is reported separated and the model is refitted without it.

Units differ across the nested models: a rating is standardised over the answers of the model's own population (all answers, those with a substance rating, those rated on all five), so the same predictor's coefficient is per a slightly different SD in each model; compare signs and intervals across models, not sizes.

Repairs are rare: a founder repair turn comes between the answer and the panel's next turn for 2 of 1388 answers (0.1%), in 2 pitches, 1 of them after a hard question. The responsiveness model's `repair_before_next` rests on these and, under hard questions, usually has no interval.

- **support** (evid, uncert): population: all answers (rated on checkable support and admitted uncertainty); 1250 answers (all questions, the largest of its outcomes), 191 on hard questions; up to 97 informative pitches.
- **substance** (evid, uncert, subs): population: answers with a substance rating; 1110 answers (all questions, the largest of its outcomes), 162 on hard questions; up to 97 informative pitches.
- **full** (evid, uncert, subs, coh, ctx): population: answers rated on support, uncertainty, substance, coherence and context; 1079 answers (all questions, the largest of its outcomes), 151 on hard questions; up to 96 informative pitches.
- **responsiveness** (uncert, repair_before_next): population: all answers (rated on admitted uncertainty); 1250 answers (all questions, the largest of its outcomes), 191 on hard questions; up to 97 informative pitches.

H9 is an exploratory pattern, not a finding: about 120 coefficients of interest are estimated with no multiple-test correction, and requiring the intervals for all and for hard questions to agree reduces false positives but does not remove them.

The context ratings (PRES-CTX in C2p, `pres_ctx`; PREP-CTX in C3 and H9's full model, `a_ctx_mean`, `ctx`) are Gemini only, not human-coded: they are not on the human coders' sheet.

Directions claimed (1 of 72 predictor rows):
- full / further_questions_same_frame / subs: positive, against the hypothesis, which expects fewer consecutive questions in the same frame (less pressing on) after a more prepared answer: here the panel asks more consecutive questions in the same frame: it presses on, more work for the panel, not less.

Not claimed: the all-questions interval excludes zero but the hard-questions interval does not (or there is none) in 13 rows:
- support / next_accept / evid: all questions +0.46 [+0.10, +0.86]; hard questions +0.67 [-0.54, +2.39].
- support / next_challenge / evid: all questions -0.91 [-1.51, -0.33]; hard questions +0.31 [-0.76, +1.38].
- support / next_question_sev / evid: all questions -0.12 [-0.20, -0.04]; hard questions -0.21 [-0.54, +0.15].
- substance / next_accept / evid: all questions +0.54 [+0.10, +1.02]; hard questions +1.12 [-0.30, +3.31].
- substance / next_challenge / evid: all questions -1.02 [-1.70, -0.40]; hard questions +0.82 [-0.29, +2.44].
- substance / next_question_sev / evid: all questions -0.13 [-0.22, -0.04]; hard questions -0.06 [-0.39, +0.30].
- substance / further_questions_same_frame / subs: all questions +0.16 [+0.01, +0.33]; hard questions +0.09 [-0.19, +0.38].
- full / next_accept / evid: all questions +0.52 [+0.04, +0.99]; hard questions +1.14 [-0.25, +3.56].
- full / next_challenge / evid: all questions -0.82 [-1.50, -0.13]; hard questions no interval.
- full / next_challenge / ctx (Gemini only, not human-coded): all questions +0.40 [+0.00, +0.70]; hard questions no interval.
- full / next_question_same_frame / ctx (Gemini only, not human-coded): all questions -0.19 [-0.33, -0.05]; hard questions -0.52 [-1.17, +0.03].
- full / next_question_sev / evid: all questions -0.11 [-0.20, -0.02]; hard questions -0.04 [-0.44, +0.35].
- full / further_questions_same_frame / ctx (Gemini only, not human-coded): all questions -0.12 [-0.23, -0.03]; hard questions -0.17 [-0.45, +0.19].

| Model | Outcome | Predictor | All questions | n (all) | Hard questions | n (hard) | Direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| support | next_accept | evid | +0.46 [+0.10, +0.86] | 953 | +0.67 [-0.54, +2.39] | 75 | not distinguishable |
| support | next_accept | uncert | +0.36 [-0.93, +1.27] | 953 | -0.41 — | 75 | not distinguishable |
| support | next_challenge | evid | -0.91 [-1.51, -0.33] | 618 | +0.31 [-0.76, +1.38] | 96 | not distinguishable |
| support | next_challenge | uncert | +0.22 [-1.48, +1.57] | 618 | -0.59 — | 96 | not distinguishable |
| support | next_further_question | evid | -0.02 [-0.36, +0.32] | 1176 | -0.51 [-1.38, +0.26] | 157 | not distinguishable |
| support | next_further_question | uncert | -0.34 [-1.07, +0.76] | 1176 | +0.51 [-0.72, +2.11] | 157 | not distinguishable |
| support | next_question_same_frame | evid | -0.02 [-0.29, +0.28] | 1071 | +0.32 [-0.42, +1.35] | 127 | not distinguishable |
| support | next_question_same_frame | uncert | +0.37 [-0.47, +1.32] | 1071 | +0.08 [-1.27, +1.76] | 127 | not distinguishable |
| support | next_question_same_frame | q_promotion | +0.48 [+0.21, +0.77] | 1071 | +0.26 [-0.61, +1.10] | 127 | covariate, no direction |
| support | next_question_sev | evid | -0.12 [-0.20, -0.04] | 1075 | -0.21 [-0.54, +0.15] | 159 | not distinguishable |
| support | next_question_sev | uncert | +0.16 [-0.21, +0.56] | 1075 | -0.07 [-0.71, +0.65] | 159 | not distinguishable |
| support | further_questions_same_frame | evid | +0.11 [-0.17, +0.41] | 1250 | +0.31 [-0.29, +0.92] | 191 | not distinguishable |
| support | further_questions_same_frame | uncert | -0.25 [-0.95, +0.25] | 1250 | -0.03 [-0.65, +0.62] | 191 | not distinguishable |
| support | further_questions_same_frame | q_promotion | +0.52 [+0.20, +0.88] | 1250 | -0.05 [-0.69, +0.59] | 191 | covariate, no direction |
| substance | next_accept | evid | +0.54 [+0.10, +1.02] | 806 | +1.12 [-0.30, +3.31] | 59 | not distinguishable |
| substance | next_accept | uncert | +0.37 [-1.11, +1.46] | 806 | -0.10 — | 59 | not distinguishable |
| substance | next_accept | subs | +0.01 [-0.19, +0.21] | 806 | -0.03 [-0.89, +0.91] | 59 | not distinguishable |
| substance | next_challenge | evid | -1.02 [-1.70, -0.40] | 532 | +0.82 [-0.29, +2.44] | 67 | not distinguishable |
| substance | next_challenge | uncert | -0.16 — | 532 | -0.30 — | 67 | not distinguishable |
| substance | next_challenge | subs | -0.21 [-0.52, +0.09] | 532 | -0.41 [-1.40, +0.70] | 67 | not distinguishable |
| substance | next_further_question | evid | -0.05 [-0.44, +0.33] | 1000 | -0.98 [-2.33, -0.06] | 112 | not distinguishable |
| substance | next_further_question | uncert | -0.21 [-1.02, +0.95] | 1000 | +0.27 [-1.44, +2.03] | 112 | not distinguishable |
| substance | next_further_question | subs | +0.05 [-0.14, +0.23] | 1000 | +0.16 [-0.58, +0.93] | 112 | not distinguishable |
| substance | next_question_same_frame | evid | -0.05 [-0.39, +0.31] | 942 | +0.35 [-0.65, +1.62] | 102 | not distinguishable |
| substance | next_question_same_frame | uncert | +0.44 [-0.48, +1.55] | 942 | -0.52 [-2.11, +1.12] | 102 | not distinguishable |
| substance | next_question_same_frame | subs | +0.06 [-0.10, +0.22] | 942 | -0.17 [-0.77, +0.43] | 102 | not distinguishable |
| substance | next_question_same_frame | q_promotion | +0.48 [+0.16, +0.79] | 942 | +0.38 [-0.69, +1.48] | 102 | covariate, no direction |
| substance | next_question_sev | evid | -0.13 [-0.22, -0.04] | 958 | -0.06 [-0.39, +0.30] | 139 | not distinguishable |
| substance | next_question_sev | uncert | +0.03 [-0.37, +0.45] | 958 | -0.39 [-0.98, +0.40] | 139 | not distinguishable |
| substance | next_question_sev | subs | +0.02 [-0.03, +0.06] | 958 | -0.10 [-0.25, +0.04] | 139 | not distinguishable |
| substance | further_questions_same_frame | evid | +0.03 [-0.26, +0.35] | 1110 | +0.30 [-0.41, +1.01] | 162 | not distinguishable |
| substance | further_questions_same_frame | uncert | -0.25 [-1.12, +0.36] | 1110 | -0.08 [-0.89, +0.77] | 162 | not distinguishable |
| substance | further_questions_same_frame | subs | +0.16 [+0.01, +0.33] | 1110 | +0.09 [-0.19, +0.38] | 162 | not distinguishable |
| substance | further_questions_same_frame | q_promotion | +0.55 [+0.21, +0.93] | 1110 | -0.03 [-0.67, +0.66] | 162 | covariate, no direction |
| full | next_accept | evid | +0.52 [+0.04, +0.99] | 780 | +1.14 [-0.25, +3.56] | 58 | not distinguishable |
| full | next_accept | uncert | +0.51 [-0.96, +1.60] | 780 | -0.18 — | 58 | not distinguishable |
| full | next_accept | subs | -0.01 [-0.28, +0.26] | 780 | -0.21 [-2.07, +1.01] | 58 | not distinguishable |
| full | next_accept | coh | +0.10 [-0.21, +0.48] | 780 | +0.14 [-1.20, +2.10] | 58 | not distinguishable |
| full | next_accept | ctx (Gemini only, not human-coded) | -0.19 [-0.48, +0.03] | 780 | +0.02 [-0.85, +0.77] | 58 | not distinguishable |
| full | next_challenge | evid | -0.82 [-1.50, -0.13] | 502 | +3.01 — | 58 | not distinguishable |
| full | next_challenge | uncert | -0.07 — | 502 | +0.20 — | 58 | not distinguishable |
| full | next_challenge | subs | -0.17 [-0.62, +0.26] | 502 | +1.49 — | 58 | not distinguishable |
| full | next_challenge | coh | -0.28 [-0.65, +0.09] | 502 | -4.40 — | 58 | not distinguishable |
| full | next_challenge | ctx (Gemini only, not human-coded) | +0.40 [+0.00, +0.70] | 502 | +2.45 — | 58 | not distinguishable |
| full | next_further_question | evid | -0.11 [-0.51, +0.28] | 958 | -1.23 [-3.30, -0.17] | 102 | not distinguishable |
| full | next_further_question | uncert | -0.21 [-1.07, +1.00] | 958 | +0.32 [-2.45, +2.37] | 102 | not distinguishable |
| full | next_further_question | subs | +0.02 [-0.24, +0.28] | 958 | -0.10 [-1.53, +1.14] | 102 | not distinguishable |
| full | next_further_question | coh | +0.09 [-0.18, +0.35] | 958 | +0.50 [-0.47, +2.60] | 102 | not distinguishable |
| full | next_further_question | ctx (Gemini only, not human-coded) | +0.02 [-0.18, +0.25] | 958 | -0.49 [-1.34, +0.05] | 102 | not distinguishable |
| full | next_question_same_frame | evid | -0.06 [-0.41, +0.31] | 920 | -0.03 [-1.42, +1.34] | 96 | not distinguishable |
| full | next_question_same_frame | uncert | +0.34 [-0.54, +1.46] | 920 | -0.78 [-3.13, +1.67] | 96 | not distinguishable |
| full | next_question_same_frame | subs | +0.15 [-0.08, +0.38] | 920 | +0.34 [-0.38, +1.44] | 96 | not distinguishable |
| full | next_question_same_frame | coh | -0.07 [-0.30, +0.16] | 920 | -0.42 [-1.47, +0.25] | 96 | not distinguishable |
| full | next_question_same_frame | ctx (Gemini only, not human-coded) | -0.19 [-0.33, -0.05] | 920 | -0.52 [-1.17, +0.03] | 96 | not distinguishable |
| full | next_question_same_frame | q_promotion | +0.52 [+0.19, +0.85] | 920 | +0.51 [-0.53, +1.62] | 96 | covariate, no direction |
| full | next_question_sev | evid | -0.11 [-0.20, -0.02] | 932 | -0.04 [-0.44, +0.35] | 130 | not distinguishable |
| full | next_question_sev | uncert | -0.02 [-0.44, +0.39] | 932 | -0.41 [-1.09, +0.34] | 130 | not distinguishable |
| full | next_question_sev | subs | +0.01 [-0.04, +0.07] | 932 | -0.15 [-0.31, +0.02] | 130 | not distinguishable |
| full | next_question_sev | coh | -0.01 [-0.07, +0.06] | 932 | +0.04 [-0.23, +0.30] | 130 | not distinguishable |
| full | next_question_sev | ctx (Gemini only, not human-coded) | +0.02 [-0.04, +0.08] | 932 | +0.10 [-0.09, +0.29] | 130 | not distinguishable |
| full | further_questions_same_frame | evid | +0.04 [-0.25, +0.36] | 1079 | +0.23 [-0.56, +0.98] | 151 | not distinguishable |
| full | further_questions_same_frame | uncert | -0.35 [-1.28, +0.25] | 1079 | -0.08 [-0.86, +0.85] | 151 | not distinguishable |
| full | further_questions_same_frame | subs | +0.27 [+0.03, +0.58] | 1079 | +0.48 [+0.05, +0.96] | 151 | positive |
| full | further_questions_same_frame | coh | -0.16 [-0.39, +0.05] | 1079 | -0.53 [-0.96, -0.01] | 151 | not distinguishable |
| full | further_questions_same_frame | ctx (Gemini only, not human-coded) | -0.12 [-0.23, -0.03] | 1079 | -0.17 [-0.45, +0.19] | 151 | not distinguishable |
| full | further_questions_same_frame | q_promotion | +0.57 [+0.21, +0.99] | 1079 | +0.00 [-0.64, +0.66] | 151 | covariate, no direction |
| responsiveness | next_accept | uncert | +0.20 [-1.05, +1.05] | 953 | -0.51 — | 75 | not distinguishable |
| responsiveness | next_accept | repair_before_next | — | 953 | — | 75 | not distinguishable |
| responsiveness | next_challenge | uncert | +0.46 [-1.14, +1.71] | 618 | -0.63 — | 96 | not distinguishable |
| responsiveness | next_challenge | repair_before_next | — | 618 | — | 96 | not distinguishable |
| responsiveness | next_further_question | uncert | -0.34 [-1.06, +0.76] | 1176 | +0.59 [-0.48, +2.13] | 157 | not distinguishable |
| responsiveness | next_further_question | repair_before_next | — | 1176 | — | 157 | not distinguishable |
| responsiveness | next_question_same_frame | uncert | +0.38 [-0.48, +1.36] | 1071 | +0.03 [-1.26, +1.63] | 127 | not distinguishable |
| responsiveness | next_question_same_frame | repair_before_next | separated | 1071 | — | 127 | not distinguishable |
| responsiveness | next_question_same_frame | q_promotion | +0.48 [+0.21, +0.77] | 1071 | +0.24 [-0.61, +1.05] | 127 | covariate, no direction |
| responsiveness | next_question_sev | uncert | +0.20 [-0.18, +0.60] | 1075 | -0.04 [-0.70, +0.66] | 159 | not distinguishable |
| responsiveness | next_question_sev | repair_before_next | +1.78 — | 1075 | +2.00 — | 159 | not distinguishable |
| responsiveness | further_questions_same_frame | uncert | -0.28 [-1.07, +0.24] | 1250 | -0.08 [-0.67, +0.57] | 191 | not distinguishable |
| responsiveness | further_questions_same_frame | repair_before_next | -0.53 — | 1250 | -0.06 — | 191 | not distinguishable |
| responsiveness | further_questions_same_frame | q_promotion | +0.52 [+0.20, +0.88] | 1250 | -0.06 [-0.72, +0.59] | 191 | covariate, no direction |

The exchange-based outcomes (next_question_same_exchange, further_questions_in_exchange) were computed as an unreported sensitivity with 499 draws and are in h9_exchange_sensitivity.csv.

The pitch-level path (answer preparedness, then questioning effort, then the offer) is described, not tested as a mediation: 100 pitches cannot carry an indirect-effect test.

## 6. H4 and H5

### H4: what the panel does after an answer

Whole encounter, no history cut; descriptive, never a predictor. Shares of answers followed by each first reaction of the panel, with and without checkable support, with 95% intervals from resampling pitches.

| Questions | Panel's first reaction | With support | Without support | Difference |
| --- | --- | --- | --- | --- |
| all | receipt | 0.14 [0.11, 0.17] (n 863) | 0.09 [0.07, 0.12] (n 682) | +0.05 [+0.02, +0.08] |
| all | question | 0.76 [0.73, 0.79] (n 863) | 0.70 [0.66, 0.74] (n 682) | +0.06 [+0.01, +0.11] |
| all | challenge | 0.04 [0.03, 0.06] (n 863) | 0.08 [0.06, 0.11] (n 682) | -0.04 [-0.07, -0.01] |
| hard | receipt | 0.15 [0.08, 0.23] (n 120) | 0.08 [0.04, 0.13] (n 190) | +0.07 [-0.02, +0.15] |
| hard | question | 0.63 [0.54, 0.73] (n 120) | 0.64 [0.58, 0.70] (n 190) | -0.01 [-0.13, +0.11] |
| hard | challenge | 0.12 [0.05, 0.21] (n 120) | 0.11 [0.07, 0.16] (n 190) | +0.01 [-0.07, +0.11] |

### H5: frame matching

Descriptive, whole encounter (1668 answers). 1429 of 1536 answers with a framed question and a framed answer keep the question's frame (93%); 107 switch. Frames derive from concepts, so matching is partly built in; deal-terms and other questions carry no frame (h5_frame_match.csv).

## 7. Self-consistency

model self-consistency under sampling; not validity: two identical passes of the model compared, over 100 videos. Stable under re-sampling is not the same as right.

| part | field | compared | agree | share_agreeing | within_one | share_within_one | what_is_counted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Outcomes and decisions | Whether the pitch got any offer | 100 | 100 | 1.0 | — | — | pitches |
| Outcomes and decisions | Whether each investor ever offered | 501 | 494 | 0.99 | — | — | investors |
| Outcomes and decisions | Decisions found by both passes | 743 | 678 | 0.91 | — | — | decisions of the primary pass |
| Outcomes and decisions | Decisions of the replicate pass also found in the primary | 739 | 678 | 0.92 | — | — | decisions of the replicate pass |
| Outcomes and decisions | Who is part of each offer | 119 | 93 | 0.78 | — | — | offers |
| Venture context | Venture context (industry, form, customer, team gender) | 400 | 378 | 0.94 | — | — | fields |
| Venture context | Primary industry (primary_industry) | 100 | 94 | 0.94 | — | — | pitches |
| Venture context | Product form (product_form) | 100 | 95 | 0.95 | — | — | pitches |
| Venture context | Customer type (customer_type) | 100 | 90 | 0.9 | — | — | pitches |
| Venture context | Presenting team gender composition (founder_gender) | 100 | 99 | 0.99 | — | — | pitches |
| Prepared presentation | Prepared presentation window (seconds in both passes' windows) | 9346 | 8875 | 0.95 | — | — | seconds in either pass's window |
| Turns | Turns found by both passes with the same function | 6791 | 4673 | 0.69 | — | — | turns of the primary pass |
| Turns | Turn function, among turns paired across passes | 5195 | 4673 | 0.9 | — | — | paired turns |
| Turn codes | Question concept (PR-CONCEPT) | 1229 | 1004 | 0.82 | — | — | paired turns with a value in both passes |
| Turn codes | Question frame (derived from the concept) | 1229 | 1066 | 0.87 | — | — | paired turns with a value in both passes |
| Turn codes | Question severity (PR-SEV) | 1229 | 1075 | 0.87 | 1195 | 0.97 | paired turns with a value in both passes |
| Turn codes | Answer coherence (PREP-COH) | 1119 | 738 | 0.66 | 1087 | 0.97 | paired turns with a value in both passes |
| Turn codes | Answer contextualisation (PREP-CTX) | 1108 | 973 | 0.88 | 1038 | 0.94 | paired turns with a value in both passes |
| Turn codes | Answer substance (PREP-SUBS) | 1201 | 760 | 0.63 | 1169 | 0.97 | paired turns with a value in both passes |
| Turn codes | Answer concept (RS-CONCEPT) | 1443 | 1105 | 0.77 | — | — | paired turns with a value in both passes |
| Turn codes | Checkable support (RS-EVID) | 1439 | 1260 | 0.88 | — | — | paired turns with a value in both passes |
| Turn codes | Answer frame (derived from the concept) | 1443 | 1216 | 0.84 | — | — | paired turns with a value in both passes |
| Turn codes | Answer directness (RS-RESP) | 1444 | 1324 | 0.92 | — | — | paired turns with a value in both passes |
| Turn codes | Admitted uncertainty or concession (RS-UNCERT) | 1439 | 1388 | 0.96 | — | — | paired turns with a value in both passes |
| Exchanges | Exchanges paired with at least half their turns in common | 1198 | 836 | 0.7 | — | — | exchanges of the primary pass |
| Exchanges | Exchange kind, among paired exchanges | 904 | 884 | 0.98 | — | — | paired exchanges |
| Exchanges | Exchange topic, among paired substantive exchanges | 711 | 549 | 0.77 | — | — | paired substantive exchanges |
| Stage A ratings | Stage A ratings of the prepared presentation | 300 | 191 | 0.64 | 300 | 1.0 | items rated in both passes |
| Stage A ratings | Stage A: pitch positivity (pitch_positivity) | 100 | 69 | 0.69 | 100 | 1.0 | pitches rated on the item in both passes |
| Stage A ratings | Stage A: expressed enthusiasm (expressed_enthusiasm) | 100 | 63 | 0.63 | 100 | 1.0 | pitches rated on the item in both passes |
| Stage A ratings | Stage A: interpersonal warmth (interpersonal_warmth) | 100 | 59 | 0.59 | 100 | 1.0 | pitches rated on the item in both passes |

Stage A ranking stability: Over 100 videos, each ranked in 3 different batches, a video's standing in one batch correlates with its standing in another (mean over the pairs of its batches) between 0.51 (interpersonal_warmth) and 0.61 (pitch_positivity).

## 8. Limits

Read from `docs/plans/project_plan_v4.md`, section 11 (the file's sha256 5198eb53672181bb23db10ed3393f5118be7cc2ee3ccdd18646995d133dd680b); the number of pitches without an offer is this run's.

- Both passes see the outcome. This is the demand-characteristics problem that led Pollack et al. to analyse funded pitches only; we keep the unfunded pitches and rely on the cut, which removes decisions from the predictors but not from the annotator's view. A possible follow-up, not done now: coders rate content cards on a video trimmed at the cut and the outcome card on the full video.
- Perceived legitimacy, the mediator in Pollack et al., is not measured; its behavioural trace (questioning effort) stands in for it and is not the same construct.
- 27 pitches without an offer in this run (plan v4 states 28, the v3-dev8 count) bound every pitch-level interval; the sample over-represents pitches with a deal relative to the show.
- Edited footage omits time and may reorder; a reaction shot is the editor's choice; turn order is broadcast order.
- Reactions are attributed to the nearest preceding answer: a reaction that returns to an earlier answer (a shark marvelling at the price two questions later) is credited to the wrong answer. Only 33% of accepts and 46% of challenges carry a response link, so links cannot fix this.
- Replicate agreement is the model against itself; two passes can be wrong the same way. Human validation covers a ten-video sample, not the corpus.
- All 100 videos are development videos: the instrument was shaped while looking at them. A confirmatory claim needs the same specification on videos not yet looked at, which this plan does not provide.
- Shark Tank selects energetic founders and the opening pitch is rehearsed for television; the range of preparedness is narrower than in a general sample.

Limits of the method:

- The leave-one-pitch-out AUC. The base-rate adjustment removes the shift of the training base rate against the held-out pitch, but a residual leave-one-out bias against weak models can remain; the bootstrap intervals of the AUC differences are not immune to it, the permutation p is (it repeats the same leave-one-out under shuffled outcomes).
- H9 runs about 120 coefficients of interest with no multiple-test correction; a direction claimed where the intervals for all and for hard questions agree is an exploratory pattern, not a finding.

## 9. Provenance

- git commit: anonymized
- release ids: v3-dev9
- seed: 42
- n_boot: 1999
- n_perm: 500
- workers: 7 (processes for H9's bootstrap draws, by default every core but one; the draws are made before any fit, so no number depends on it)
- single_draws: 9999
- penalty: inner_cv
- passes: main reads the best record of each video; `both_passes` averages both model passes
- H9: run
- with_shark_level: True
- spec: docs/plans/design docs/2026-10-19-pitch-level-refocus.md (sha256 of the whole file afaedc5c06387d05ddd569f395674f2e240242ad971f79600f383dd252826d48)
- limits: docs/plans/project_plan_v4.md (sha256 of the whole file 5198eb53672181bb23db10ed3393f5118be7cc2ee3ccdd18646995d133dd680b)
- run: runs/full-corpus-100; videos: 100 (a selection)
- figures: listed in manifest.json (`figures`, `figures_skipped`)
- shark level (archived design, for F1; reported, not tested): tables in shark_level/; identity alone ranks the sharks of a pitch at 0.670 [0.601, 0.740] within-pitch AUC
