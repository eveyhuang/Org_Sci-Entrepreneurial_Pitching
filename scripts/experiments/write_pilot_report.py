#!/usr/bin/env python3
"""Render pilot annotations, missingness and grouped AUC without selecting a winning model."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/stage_b_structured_pilot5'
def read(p):return json.loads(p.read_text())
def esc(x):return str(x).replace('|','\\|').replace('\n',' ')
def number(x):return 'not estimable' if x is None else f'{round(x,12):.3f}'
def main():
 d=read(RUN/'analysis.json');out=RUN/'report.md'
 lines=['# Five-video structured Stage A/B pilot','', '**This is a development pilot with five independent pitch clusters. AUC describes exploratory discrimination against model-reviewed on-air offer labels; it does not validate annotation accuracy or establish a causal effect.**','', '## Sample and pipeline','', 'Videos were sampled with seed 20260925, without outcome balancing, excluding Copy Keyboard. One obvious quiz/compilation was excluded by title before outcome review. The workflow ran fresh full-video timeline and segmentation, a separate decision/boundary review, Q&A timeline repair/review, issue-map review and freeze gate, Stage A v2 prepared-delivery ratings, all twenty episode codes, pitch consistency and conditional visual-coverage checks.','', '| Video | Prepared window (s) | Pre-decision cutoff (s) | Episodes | Q&A turns | Offers / out / unknown | Positivity | Consistency |','|---|---|---|---|---|---|---|---|']
 summary=[
  '', '## Main findings', '',
  'All five videos completed: **40 episodes, 307 reviewed Q&A turns, 800 episode-code cells and 25 shark outcomes (6 offers, 19 no-offer exits)**. The final artifact integrity audit passed for all five. Outcomes and semantic ratings are model-reviewed, not human gold.',
  '',
  '**Predictive result:** the prespecified founder-interaction model has pooled leave-one-video-out AUC **0.360** and mean within-video AUC **0.292**. Adding prepared positivity changes neither result; adding probe controls gives **0.272 / 0.188**. This pilot does not demonstrate useful predictive discrimination. It also does not establish that interaction has no relationship with decisions: only five pitch clusters, sparse behavior, missing exposure and unresolved segmentation issues are represented.',
  '',
  '**The baseline comparison is uninformative here:** all five positivity scores are 5/5. Raw positivity AUC is **0.500**. The fitted baseline’s pooled AUC of **0.281** is solely a leave-one-video-out class-prior artifact, not negative predictive information in positivity. Its within-video AUC is 0.500.',
  '',
  '**Measurement findings:** 10 episode outputs were rejected for missing evidence references and recoded. The visibility gate rejected 11 proposed affect ratings, leaving investor affect uncodable in **39/40** episodes. Recovery is uncodable in **40/40**; coherence in **22/40**; substance in **10/40**. These missing values must not be converted into negative behavior or poor quality.',
  '',
  '**Remaining problems:** uptake is uncodable in 17/40 episodes, partly because the map cannot reference the next investor move across an episode boundary. Slate’s procedural tasting openings raise substantive-probe eligibility concerns. Five sharks, including two offer-makers, have no opening episode. Coherence is 4 in 17 of its 18 codable cases; repair acknowledgment and modification occur only once each, and uncertainty acknowledgment never occurs. These features have little variation in this sample.',
  '',
  '**Fixed during the run:** a prepared/Q&A turn-ID collision in Wendy’s consistency prompt. The affected pass was rerun; all five final consistency counts are zero. Wendy and Slate include comparable-claim checks, while the other three report no comparable repeated claims. Those zero counts therefore do not all mean the same evidential thing.',
  '',
  '**Judgment:** the revisions improve traceability and detect structural errors, but the instrument is not yet ready for an unattended 100-video research run. Prioritize cross-boundary uptake context, consistent substantive-probe eligibility, dyadic follow-up attribution and human adjudication. No codebook definition was changed to improve AUC.',
 ]
 lines[4:4]=summary+['']
 long=[]
 for v in d['videos']:
  lines.append('| '+' | '.join(map(esc,[v['title'],f"{v['prepared_start']}–{v['prepared_end']}",v['cutoff'],v['episodes'],v['qa_turns'],f"{v['offers']} / {v['out']} / {v['unknown']}",v['positivity'],v['pitch_consistency']]))+' |')
  r=RUN/v['video_id'];b=read(r/'frozen_bundle.json');ts={t['turn_idx']:t for t in b['timeline']['turns']};detail=['# '+v['title'],'','Model-reviewed development annotation; no independent human gold.','']
  for e in b['mapping']['episodes']:
   rec=read(r/f"results/accepted_{e['episode_id']}.json")['record'];start=min(ts[i]['start_s'] for i in e['turn_refs']);end=max(ts[i]['end_s'] for i in e['turn_refs'])
   detail += [f"## {e['episode_id']}: {e['topic']} ({start}–{end} s)",'',f"Probe {e['opening_probe']}; initial answer {e['initial_answer']}; uptake {e['first_uptake']} ({e['uptake_status']}).",e['rationale'],'','| Code | Value | Evidence refs | Evidence |','|---|---|---|---|']
   for code,x in rec.items():
    val='uncodable' if x['uncodable'] else x['value'];detail.append('| '+' | '.join(map(esc,[code,val,x['turn_refs'],x['evidence']]))+' |');long.append({'video_id':v['video_id'],'episode_id':e['episode_id'],'opening_shark':e['probing_speaker'],'start_s':start,'end_s':end,'code':code,'value':x['value'],'uncodable':x['uncodable'],'turn_refs':json.dumps(x['turn_refs']),'evidence':x['evidence']})
  (r/'annotation.md').write_text('\n'.join(detail)+'\n')
 lines+=['','## Grouped out-of-sample AUC','', 'Outcome = a shark ever makes or joins a concrete on-air offer, even if rejected or later withdrawn. An observed exit without offer is negative; unknown decisions are excluded. Train on four videos and predict the fifth, repeating five times. All sharks and episodes from a held-out video stay out of training. Fixed ridge logistic regression (L2=1), no feature/hyperparameter search. Imputation and standardization use training videos only.','', '| Prespecified model | Pooled out-of-fold AUC | Mean within-video AUC where defined |','|---|---|---|']
 for name,m in d['models'].items():lines.append(f"| {name} | {number(m['pooled_oof_auc'])} | {number(m['mean_within_video_auc'])} |")
 diagnostic=d['diagnostics']['exposure_only_not_prespecified']
 lines+=['',f"Additional coverage diagnostic (not one of the four prespecified models): episode count alone gives pooled AUC {number(diagnostic['pooled_oof_auc'])}, mean within-video AUC {number(diagnostic['mean_within_video_auc'])}. This checks whether speaking opportunities, rather than coded response qualities, may contribute to discrimination; it is not a model-selection result."]
 lines+=['','Founder features are response directness, evidence rate, substance, coherence and self-repair, averaged over episodes opened by that shark. Interaction models also include observed episode count as an exposure control. The baseline is Stage A positivity; the conditional model adds probe severity and promotion share. Episode-wide repair describes the opener-associated issue, not necessarily repair directed exclusively at that investor. Investor uptake/affect, dyadic rapport, offer terms and post-decision behavior are excluded from primary predictors.','', 'A positivity-only score is identical across sharks in a held-out pitch, so its within-pitch AUC is 0.5 when both classes occur. Pooled leave-one-video-out AUC can also reflect calibration differences between tiny training folds; examine both summaries. These models omit the full study\'s venture controls, shark fixed effects and pitch random effect.','', '| Held-out video | Test sharks | Offers | Baseline AUC | Interaction AUC | Combined AUC | Conditional AUC |','|---|---|---|---|---|---|---|']
 for v in d['videos']:
  values=[]
  for m in d['models'].values():values.append(number(next(f for f in m['folds'] if f['held_out_video']==v['video_id'])['within_video_auc']))
  lines.append('| '+' | '.join(map(esc,[v['title'],v['offers']+v['out'],v['offers']]+values))+' |')
 lines+=['','## Segmentation and runtime checks','', '| Video | Initial → reviewed episodes | Uptake states | Episode outputs rejected by evidence gate | Failed provider attempts |','|---|---|---|---|---|']
 for v in d['videos']:
  lines.append('| '+' | '.join(map(esc,[v['title'],f"{v['initial_episode_count']} → {v['episodes']}",v['uptake_status_counts'],v['episode_qc_rejected_outputs'],v['provider_failed_attempts']]))+' |')
 lines+=['','Counts compare two stages of this revised workflow, not old versus new codebooks. Episode gates check structural eligibility; passing does not establish semantic correctness. Failed provider attempts include preserved pre-resume failures. Historical failure files remain as audit evidence even after completion.','', '## Codability','', '| Code | Uncodable / episode cells | Fraction |','|---|---|---|']
 for code,m in d['code_missingness'].items():lines.append(f"| {code} | {m['uncodable']} / {m['n']} | {m['uncodable_fraction']:.1%} |")
 no_exposure=sum(r['n_episodes']==0 for r in d['decision_rows']);lines+=['',f"Sharks with no opening episode before cutoff: {no_exposure}/{len(d['decision_rows'])}. Their interaction features are missing, not observed zeros; the model uses training-only imputation plus exposure count.",'','## Audit and usage','']
 for v in d['videos']:lines += [f"- **{v['title']}**: {v['calls']} successful generations; {v['gate_rejections']} timeline/map gate rejections. [Full annotation]({v['video_id']}/annotation.md)."]
 lines += ['',f"Total successful generations: {d['api_calls']}; tokens: {d['usage']}. Approximate cost at [standard Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing), $2/M input and $12/M output including thinking: ${d['cost_at_2_12_usd_per_million']:.2f}. Unreported usage on provider failures is not included.",'', '[Analysis plan](analysis_plan.md), [machine-readable analysis](analysis.json), [shark-level feature table](decision_features.csv), [out-of-fold predictions](oof_predictions.csv), [episode evidence table](episode_features.csv).','', 'Final verification: all five artifact integrity audits passed; 25 focused tests passed (AUC, recovery behavior and structured gates). [Integrity audit](integrity_audit.json). Independent human review remains necessary for turn completeness, speaker attribution, boundaries, semantic code judgments and outcome labels. No result should be described as a validated AUC on Shark Tank generally.']
 with (RUN/'episode_features.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(long[0]));w.writeheader();w.writerows(long)
 notes=(RUN/'pilot_review_notes.md').read_text()
 lines+=['','## Detailed measurement findings','',notes.replace('# Annotation audit notes (before AUC results)','').replace('## ','### ') ]
 out.write_text('\n'.join(lines)+'\n');print(out)
if __name__=='__main__':main()
