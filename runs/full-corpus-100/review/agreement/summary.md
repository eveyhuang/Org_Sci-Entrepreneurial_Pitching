# Human review summary: full-corpus-100

10 videos with an imported sheet, 640 reviewed rows; 23 structural notes, 0 outcome overrides, 13 first-decision overrides, 26 rows without a definite value.

## Sampled rows (the estimate)

Rows drawn at random, pooled over videos: the share on which the human gives the primary pass's value, with a 95% interval, beside the pass-to-pass figure on the rows that have a replicate value. The first decision agrees when both times map to the same turn of the sheet’s reference pass; unmappable times are excluded and counted separately. Other clock values count as the same within tolerance (a decision time within the agreement settings' tolerance, each edge of the prepared window within two seconds); ordinal fields (rated scales) are compared exactly, and the `within one` column gives the share within one scale point. Rows of one video are not independent, so with at least 5 videos carrying rows of a class the interval resamples whole videos (cluster bootstrap, seeded); with fewer it is the Wilson interval, labelled `Wilson; treats rows as independent; optimistic`.

| field class | videos | rows | human = primary | 95% interval | human = replicate | primary = replicate | within one (human, primary) | interval method |
|---|---|---|---|---|---|---|---|---|
| turn_code | 10 | 454 | 0.63 (284/454) | 0.56-0.70 | 0.63 (248/394) | 0.76 (154/202) | 0.86 (198/229) | cluster bootstrap over videos (1,999 draws, seed 0) |
| stage_a | 10 | 60 | 0.25 (15/60) | 0.15-0.35 | 0.33 (20/60) | 0.70 (21/30) | 0.73 (44/60) | cluster bootstrap over videos (1,999 draws, seed 0) |
| prep_pres | 10 | 60 | 0.57 (34/60) | 0.47-0.65 | 0.53 (32/60) | 0.63 (19/30) | 1.00 (60/60) | cluster bootstrap over videos (1,999 draws, seed 0) |
| venture | 10 | 20 | 1.00 (20/20) | 1.00-1.00 | 1.00 (20/20) | 1.00 (10/10) | - | cluster bootstrap over videos (1,999 draws, seed 0) |
| cut | 10 | 20 | 0.80 (8/10) | 0.40-1.00 | 0.75 (6/8) | 1.00 (8/8) | - | cluster bootstrap over videos (1,999 draws, seed 0) |

## Agreement per code, coder against model (the validation table)

Each reported code with the statistics of its kind (validation_reporting.yaml), from the coder-coded videos: raw agreement with its 95% interval resampling videos, Krippendorff's alpha in parentheses with its interval, alpha's Landis & Koch band, and flags in words; the column without unsure units is a point. `--paper-table` adds coder against coder, pass against pass, the appendix and the note on the statistics.

| Field | Kind | Units (videos) | Agreement (α) | Band of α | Flags | Without unsure: point (n) |
|---|---|---|---|---|---|---|
| PR-FRAME | categorical | 37 (10); 72 judgements | 0.83 [0.75, 0.91] (α 0.67 [0.48, 0.82]) | substantial | — | 0.83 (64) |
| RS-RESP | categorical | 40 (10); 76 judgements | 0.76 [0.59, 0.89] (α 0.25 [-0.10, 0.52]) | fair | prevalence: read AC1 (0.74, substantial) rather than α | 0.79 (71) |
| RS-EVID | binary | 40 (10); 77 judgements | 0.82 [0.72, 0.91] (α 0.64 [0.44, 0.82]) | substantial | — | 0.86 (69) |
| PR-SEV | ordinal | 39 (10); 74 judgements | identical 0.76 [0.64, 0.86]; within one 0.96 [0.92, 1.00] (α 0.57 [0.37, 0.79]) | moderate | prevalence: read AC2 (0.85, almost perfect) rather than α | 0.78 (69) |
| PREP-SUBS | ordinal | 37 (10); 71 judgements | identical 0.32 [0.22, 0.42]; within one 0.86 [0.73, 0.95] (α 0.27 [0.00, 0.50]) | fair | — | 0.32 (66) |
| PREP-COH | ordinal | 37 (10); 72 judgements | identical 0.47 [0.33, 0.61]; within one 0.92 [0.86, 0.97] (α 0.24 [-0.05, 0.41]) | fair | — | 0.45 (65) |
| PRES-SUBS | ordinal | 10 (10); 20 judgements | identical 0.40 (8/20) [0.20, 0.60]; within one 1.00 (20/20) [1.00, 1.00] (α 0.38 [-0.21, 0.76]) | fair | small n (10 units) | 0.42 (19) |
| PRES-COH | ordinal | 10 (10); 20 judgements | identical 0.75 (15/20) [0.55, 0.95]; within one 1.00 (20/20) [1.00, 1.00] (α 0.61 [-0.05, 0.90]) | substantial | prevalence: read AC2 (0.98, almost perfect) rather than α; small n (10 units) | 0.78 (18) |
| PRES-EVID | ordinal | 10 (10); 20 judgements | identical 0.55 (11/20) [0.30, 0.80]; within one 1.00 (20/20) [1.00, 1.00] (α 0.65 [0.15, 0.84]) | substantial | small n (10 units) | 0.53 (17) |
| pitch_positivity | ordinal | 10 (10); 20 judgements | identical 0.30 (6/20) [0.10, 0.50]; within one 0.75 (15/20) [0.60, 0.90] (α -0.42 [-0.61, -0.32]) | poor | small n (10 units) | 0.33 (18) |
| expressed_enthusiasm | ordinal | 10 (10); 20 judgements | identical 0.25 (5/20) [0.10, 0.40]; within one 0.70 (14/20) [0.55, 0.85] (α -0.25 [-0.55, -0.02]) | poor | small n (10 units) | 0.22 (18) |
| interpersonal_warmth | ordinal | 10 (10); 20 judgements | identical 0.20 (4/20) [0.00, 0.40]; within one 0.75 (15/20) [0.60, 0.90] (α -0.04 [-0.53, 0.35]) | poor | small n (10 units) | 0.21 (19) |
| cut | clock | 10 (10); 20 judgements | same turn 0.80 (8/10) [0.37, 1.00]; median 12.0 s; 10 undetermined | — | small n (10 units) | 0.80 (10) |
| venture:ask | exact | 10 (10); 20 judgements | 1.00 (20/20) [1.00, 1.00] | — | small n (10 units) | 1.00 (20) |
| pair:pitch_positivity | comparison | 25 (10); 50 judgements | 32/50 = 0.64 [0.36, 0.91]; p = 0.065 against 0.5 | — | — | — |
| pair:expressed_enthusiasm | comparison | 25 (10); 50 judgements | 36/50 = 0.72 [0.46, 0.94]; p = 0.003 against 0.5 | — | — | — |
| pair:interpersonal_warmth | comparison | 25 (10); 50 judgements | 32/50 = 0.64 [0.42, 0.85]; p = 0.065 against 0.5 | — | — | — |

## Without the rows a coder was not sure of

51 of 640 reviewed rows carry a coder's best guess ticked not sure ([unsure] in the note); the sampled figures without them:

| field class | videos | rows | human = primary | 95% interval | human = replicate | primary = replicate | within one (human, primary) | interval method |
|---|---|---|---|---|---|---|---|---|
| turn_code | 10 | 414 | 0.64 (265/414) | 0.58-0.71 | 0.65 (231/358) | 0.76 (153/201) | 0.87 (182/210) | cluster bootstrap over videos (1,999 draws, seed 0) |
| stage_a | 10 | 55 | 0.25 (14/55) | 0.13-0.36 | 0.35 (19/55) | 0.70 (21/30) | 0.71 (39/55) | cluster bootstrap over videos (1,999 draws, seed 0) |
| prep_pres | 10 | 54 | 0.57 (31/54) | 0.48-0.67 | 0.50 (27/54) | 0.63 (19/30) | 1.00 (54/54) | cluster bootstrap over videos (1,999 draws, seed 0) |
| venture | 10 | 20 | 1.00 (20/20) | 1.00-1.00 | 1.00 (20/20) | 1.00 (10/10) | - | cluster bootstrap over videos (1,999 draws, seed 0) |
| cut | 10 | 20 | 0.80 (8/10) | 0.40-1.00 | 0.75 (6/8) | 1.00 (8/8) | - | cluster bootstrap over videos (1,999 draws, seed 0) |

## By outcome stratum

The sample is stratified by the outcome (any_offer): 5 with an offer, 5 without, drawn from 69 and 26 eligible pitches of the run. A reweighted figure weights each judgement by its stratum's share of the run's eligible pitches over its share of the videos pooled here (weights 1.45 and 0.55), so it describes the run's mix; the pooled figure is the main table's, unweighted. The main table's figures are unweighted.

Each cell leads with raw agreement (identical; within one for an ordinal code; same turn for the first decision) with Krippendorff's α in parentheses for categorical, binary and ordinal codes, and the judgements (n) in the stratum. Every figure in this section is a point, without an interval. The comparisons are left out: a pair's two clips can come from both strata.

| Field | With an offer: agreement (α), point | Without an offer: agreement (α), point | Reweighted: agreement (α), point | Pooled (main table): agreement (α), point |
|---|---|---|---|---|
| PR-FRAME | 0.89 (α 0.79), n 37 | 0.77 (α 0.53), n 35 | 0.86 (α 0.73), n 72 | 0.83 (α 0.67) |
| RS-RESP | 0.85 (α 0.48), n 39 | 0.68 (α 0.04), n 37 | 0.80 (α 0.35), n 76 | 0.76 (α 0.25) |
| RS-EVID | 0.85 (α 0.69), n 40 | 0.78 (α 0.57), n 37 | 0.83 (α 0.66), n 77 | 0.82 (α 0.64) |
| PR-SEV | 0.71; within one 0.95 (α 0.40), n 38 | 0.81; within one 0.97 (α 0.76), n 36 | 0.74; within one 0.95 (α 0.49), n 74 | 0.76; within one 0.96 (α 0.57) |
| PREP-SUBS | 0.33; within one 0.82 (α 0.35), n 39 | 0.31; within one 0.91 (α 0.08), n 32 | 0.33; within one 0.84 (α 0.32), n 71 | 0.32; within one 0.86 (α 0.27) |
| PREP-COH | 0.47; within one 0.93 (α 0.28), n 40 | 0.47; within one 0.91 (α 0.19), n 32 | 0.47; within one 0.92 (α 0.26), n 72 | 0.47; within one 0.92 (α 0.24) |
| PRES-SUBS | 0.30; within one 1.00 (α 0.12), n 10 | 0.50; within one 1.00 (α 0.59), n 10 | 0.35; within one 1.00 (α 0.26), n 20 | 0.40; within one 1.00 (α 0.38) |
| PRES-COH | 0.80; within one 1.00 (α 0.05), n 10 | 0.70; within one 1.00 (α 0.81), n 10 | 0.77; within one 1.00 (α 0.43), n 20 | 0.75; within one 1.00 (α 0.61) |
| PRES-EVID | 0.40; within one 1.00 (α 0.59), n 10 | 0.70; within one 1.00 (α 0.68), n 10 | 0.48; within one 1.00 (α 0.62), n 20 | 0.55; within one 1.00 (α 0.65) |
| pitch_positivity | 0.20; within one 0.70 (α -0.34), n 10 | 0.40; within one 0.80 (α -0.45), n 10 | 0.25; within one 0.73 (α -0.40), n 20 | 0.30; within one 0.75 (α -0.42) |
| expressed_enthusiasm | 0.30; within one 0.60 (α -0.07), n 10 | 0.20; within one 0.80 (α -0.41), n 10 | 0.27; within one 0.65 (α -0.18), n 20 | 0.25; within one 0.70 (α -0.25) |
| interpersonal_warmth | 0.30; within one 0.80 (α 0.31), n 10 | 0.10; within one 0.70 (α -0.48), n 10 | 0.25; within one 0.77 (α 0.13), n 20 | 0.20; within one 0.75 (α -0.04) |
| cut | 1.00, n 4 | 0.67, n 6 | 0.88, n 20 | 0.80 |
| venture:ask | 1.00, n 10 | 1.00, n 10 | 1.00, n 20 | 1.00 |

## Coders

Each coder's own agreement with the primary pass, pooled over the videos that coder judged.

### AV

| field class | videos | rows | human = primary | 95% interval | human = replicate | primary = replicate | within one (human, primary) | interval method |
|---|---|---|---|---|---|---|---|---|
| turn_code | 10 | 227 | 0.61 (138/227) | 0.54-0.69 | 0.65 (128/198) | 0.76 (150/198) | 0.85 (97/114) | cluster bootstrap over videos (1,999 draws, seed 0) |
| stage_a | 10 | 30 | 0.17 (5/30) | 0.03-0.30 | 0.13 (4/30) | 0.70 (21/30) | 0.50 (15/30) | cluster bootstrap over videos (1,999 draws, seed 0) |
| prep_pres | 10 | 30 | 0.47 (14/30) | 0.30-0.63 | 0.47 (14/30) | 0.63 (19/30) | 1.00 (30/30) | cluster bootstrap over videos (1,999 draws, seed 0) |
| venture | 10 | 10 | 1.00 (10/10) | 1.00-1.00 | 1.00 (10/10) | 1.00 (10/10) | - | cluster bootstrap over videos (1,999 draws, seed 0) |
| cut | 10 | 10 | 0.80 (4/5) | 0.40-1.00 | 0.75 (3/4) | 1.00 (8/8) | - | cluster bootstrap over videos (1,999 draws, seed 0) |

### MX

| field class | videos | rows | human = primary | 95% interval | human = replicate | primary = replicate | within one (human, primary) | interval method |
|---|---|---|---|---|---|---|---|---|
| turn_code | 10 | 227 | 0.64 (146/227) | 0.56-0.72 | 0.61 (120/196) | 0.77 (150/196) | 0.88 (101/115) | cluster bootstrap over videos (1,999 draws, seed 0) |
| stage_a | 10 | 30 | 0.33 (10/30) | 0.17-0.50 | 0.53 (16/30) | 0.70 (21/30) | 0.97 (29/30) | cluster bootstrap over videos (1,999 draws, seed 0) |
| prep_pres | 10 | 30 | 0.67 (20/30) | 0.53-0.80 | 0.60 (18/30) | 0.63 (19/30) | 1.00 (30/30) | cluster bootstrap over videos (1,999 draws, seed 0) |
| venture | 10 | 10 | 1.00 (10/10) | 1.00-1.00 | 1.00 (10/10) | 1.00 (10/10) | - | cluster bootstrap over videos (1,999 draws, seed 0) |
| cut | 10 | 10 | 0.80 (4/5) | 0.40-1.00 | 0.75 (3/4) | 1.00 (8/8) | - | cluster bootstrap over videos (1,999 draws, seed 0) |

## Coder against coder

Rows both coders gave a definite value, on videos judged by exactly two coders; `unclear` and `structural` rows are dropped. The interval resamples whole videos from 5 videos, else it is the Wilson interval treating rows as independent.

| field class | videos | rows | coder = coder | 95% interval | within one | interval method |
|---|---|---|---|---|---|---|
| turn_code | 10 | 218 | 0.62 (136/218) | 0.54-0.70 | 0.87 (96/110) | cluster bootstrap over videos (1,999 draws, seed 0) |
| stage_a | 10 | 30 | 0.13 (4/30) | 0.03-0.23 | 0.63 (19/30) | cluster bootstrap over videos (1,999 draws, seed 0) |
| prep_pres | 10 | 30 | 0.50 (15/30) | 0.37-0.63 | 0.90 (27/30) | cluster bootstrap over videos (1,999 draws, seed 0) |
| venture | 10 | 10 | 1.00 (10/10) | 1.00-1.00 | - | cluster bootstrap over videos (1,999 draws, seed 0) |
| cut | 5 | 5 | 1.00 (5/5) | 1.00-1.00 | - | cluster bootstrap over videos (1,999 draws, seed 0) |

## Pair judgements (the clip session)

Which of two prepared presentations shows more of a Stage A item: the share of the coders' answers that agree with the model's rank scores (pooled over coders), coder against coder, and the coders' mean share in the narrowest and widest third of the model's score gaps; chance is 0.5. The 95% interval resamples videos (a pair weighted by how often each of its two videos is drawn), the interval the validation table gives; the exact (Clopper–Pearson) one is in the table's appendix.

| item | pairs | answers | coders = model | 95% interval | coder = coder | narrowest third | widest third |
|---|---|---|---|---|---|---|---|
| pitch_positivity | 25 | 50 | 0.64 | 0.36-0.91 | 0.68 | 0.61 | 0.69 |
| expressed_enthusiasm | 25 | 50 | 0.72 | 0.46-0.94 | 0.68 | 0.78 | 0.69 |
| interpersonal_warmth | 25 | 50 | 0.64 | 0.42-0.85 | 0.52 | 0.56 | 0.69 |

## Structural

Rows a reviewer marked structural (the unit itself is wrong), per field class, over the rows that measure validity.

- cut: 0 of 20 rows structural (0.00)
- prep_pres: 0 of 60 rows structural (0.00)
- stage_a: 0 of 60 rows structural (0.00)
- turn_code: 23 of 480 rows structural (0.05)
- venture: 0 of 20 rows structural (0.00)

By coder:

- AV: cut 0/10, prep_pres 0/30, stage_a 0/30, turn_code 13/240, venture 0/10
- MX: cut 0/10, prep_pres 0/30, stage_a 0/30, turn_code 10/240, venture 0/10
