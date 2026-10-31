# Review sheet: st_d13372af0c

- Video file: `/Users/eveyhuang/Documents/NICO/entrepreneurship-pitch-interaction/data/shark tank/downloads/Barbara Gets Mean With Sheets Laundry Club | Shark Tank US | Shark Tank Global.mp4`
- Run: `runs/full-corpus-100` (instrument release v3-dev9)
- Sheet: `st_d13372af0c__review.csv`, 25 rows, preset `exchange`
- Mode: blind: the model's values and the reasons for selecting a row are hidden; code each row from the footage
- Prepared presentation: this sheet does not show the window the model recorded (the coding page, built with the `page` subcommand, shows it as an aid), and no row of it checks the window

## Actors

| id | role | label |
|---|---|---|
| H1 | host | Narrator |
| F1 | founder | Chris Videau |
| F2 | founder | Chris Campbell |
| S1 | investor | Mark Cuban |
| S2 | investor | Kevin O'Leary |
| S3 | investor | Lori Greiner |
| S4 | investor | Barbara Corcoran |
| S5 | investor | Daniel Lubetzky |

## How to fill the sheet

Each row is one field of the record. `path` says which field, `clock_start`-`clock_end` where to look in the video, `context` which turn it belongs to, or for a pitch row the question it asks. Watch that part of the video and fill only the last three columns; leave every other column as it is.

The `pitch:first_decision` row belongs to the whole video: watch it to the end and give the clock at which the first decision is spoken, even if other questions follow it. Each question-answer pair carries six codes: the question's concept and severity, the answer's responsiveness, evidence, substance and coherence. Choosing `other` for a concept needs a note: name the matter in a few words (the coding page's note box asks for it, as it does when you tick a box).

## What blind mode still reveals

Blind mode hides the model's values, the reasons a row was selected and the prepared window. It cannot hide everything, and you should know what you are still shown: the model's turn segmentation (where each turn starts and ends); who speaks each turn; and, through the path names, which turns the model took for questions or answers (a row `turn:<id>/code:PR-...` is a turn coded as a question, `turn:<id>/code:RS-...` or `PREP-...` one coded as an answer). A row that exists only because the other pass lacks a unit (`unit_presence`) is left off a blind sheet for that reason. Agreement measured on a blind sheet is therefore agreement given the model's segmentation, not independent coding from the footage alone.

- `human_value`: what the footage supports, written as in `allowed_values` when that column lists values.
- `resolution`: optional when `human_value` is filled. One of:
  - `correct`: `human_value` is what the footage shows (assumed whenever `human_value` is filled). With `human_value` left empty it means the field should be empty.
  - `unclear`: the footage does not let you decide. Nothing changes; say why in `note`.
  - `structural`: the problem is not this value but the unit itself (a missing turn, a wrong boundary, two turns in one). Nothing changes; describe it in `note`.
- `note`: anything a later reader needs.

A row counts as reviewed only when `human_value` or `resolution` is filled; rows left blank are not treated as confirmed. Times are clock strings (`1:48`). A code that cannot be given a value takes `missing:<reason>` with one of: not_applicable, no_opportunity, not_observable, edited_gap, ambiguous, too_short.

## Presets

- `exchange` (this sheet): the full-video sheet, a random sample of rows per video: the pitch's first decision (when is it spoken) and a number of question-answer pairs (the question's concept and severity; the answer's responsiveness, evidence, substance and coherence), set per video or planned from --rows-per-field; exchange-topic, Stage A and venture rows only when --exchanges-per-video, --stage-a-per-video or --venture-per-video ask for them (the exchange topics are not drawn by default since 2026-10-21), and the outcome (did any investor make an offer) and the check of the prepared window only when --field-classes names `outcome` or `prepared_window` (neither is drawn by default since 2026-10-21); every row an unbiased estimate; for measuring agreement over many videos.
- `full`: the complete sheet: every flagged row of every field class plus 30 sampled ones; for adjudicating one video in depth.
- `flagged`: the flagged rows up to a cap of 60 plus 20 sampled ones; for correcting what looks wrong.
- `quick`: as flagged with a cap of 25 and 15 sampled ones; for a fast look at what looks wrong.

## Allowed values

For the kinds of row on this sheet:

- `cut` (`pitch:first_decision`): the clock (`M:SS`) of the first decision any investor speaks, an offer or an exit, or empty when there is none. A clock within five seconds of the model's is the same moment.
- `turn_code`: `other_concept` is a few words naming the matter when the concept is `other`; the codes:
  - `PR-CONCEPT` (question turns): customer_acquisition, go_to_market, market_opportunity, assets, self_promotion, sales, usage, vision, forecast, execution, safety_security, liability, competition, operating_efficiency, customer_retention, team_capabilities, deal_terms, other, NONE
  - `PR-SEV` (question turns): an integer 1-3
  - `RS-RESP` (answer turns): addresses, partial, deflects, NONE
  - `RS-CONCEPT` (answer turns): customer_acquisition, go_to_market, market_opportunity, assets, self_promotion, sales, usage, vision, forecast, execution, safety_security, liability, competition, operating_efficiency, customer_retention, team_capabilities, deal_terms, other, NONE
  - `RS-EVID` (answer turns): true, false
  - `RS-UNCERT` (answer turns): true, false
  - `PREP-SUBS` (answer turns): an integer 1-5
  - `PREP-COH` (answer turns): an integer 1-5
  - `PREP-CTX` (answer turns): an integer 1-5
