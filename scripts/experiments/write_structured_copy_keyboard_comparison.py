#!/usr/bin/env python3
"""Render the structured rerun alongside historical outputs without implying accuracy."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/stage_b_copy_keyboard_structured_test'
OLD=ROOT/'runs/stage_b_copy_keyboard_revision_test'
def read(p):return json.loads(p.read_text())
def value(v):return 'uncodable' if v['uncodable'] else str(v['value'])
def esc(s):return str(s).replace('|','\\|').replace('\n',' ')
def main():
 b=read(RUN/'frozen_bundle.json');turns={t['turn_idx']:t for t in b['timeline']['turns']};eps=b['mapping']['episodes']
 lines=['# Copy Keyboard structured rerun: returned annotations','', 'Development run only. Source review used Gemini, not independent human coders. Counts of changed cells are not accuracy measures. Historical boundaries differ, so comparisons are issue-aligned rather than identical-clip experiments.','', '## Reviewed issue map','', '| Episode | Source seconds | Probe | Initial answer | First uptake | Uptake status |','|---|---|---|---|---|---|']
 for e in eps:
  lines.append('| '+' | '.join(map(esc,[e['episode_id'],f"{min(turns[i]['start_s'] for i in e['turn_refs'])}–{max(turns[i]['end_s'] for i in e['turn_refs'])}",e['opening_probe'],e['initial_answer'],e['first_uptake'],e['uptake_status']]))+' |')
 lines+=['', 'Excluded turns: '+esc(b['mapping']['excluded']), '', '## Three-stage comparison','', 'Each cell is **original preview → wording-only rebuilt run → structured rerun**. Original E0 is compared with the merged first issue; original E1 is now a later exchange, not a separate initial-answer unit.','']
 codes=list(read(OLD/'baseline/episode_0.json')['record'])
 current=[]
 for e in eps:
  current.append(read(RUN/f"results/accepted_{e['episode_id']}.json")['record'])
 if len(eps)!=3:raise ValueError('Renderer mapping requires manual review for other episode counts')
 lines+=['| Code | Origin/title | Price/sales | Patent/defensibility |','|---|---|---|---|']
 for c in codes:
  cells=[]
  for j,i in enumerate([0,2,3]):
   old=read(OLD/f'baseline/episode_{i}.json')['record'][c];prior=read(OLD/f'results/rebuilt_episode_{j}.json')['record'][c]
   cells.append(' → '.join([value(old),value(prior),value(current[j][c])]))
  lines.append('| '+c+' | '+' | '.join(cells)+' |')
 for e,r in zip(eps,current):
  lines+=['', '## '+e['episode_id']+': '+e['topic'],'','| Code | Value | Refs | Evidence |','|---|---|---|---|']
  for c,v in r.items():lines.append('| '+' | '.join(map(esc,[c,value(v),v['turn_refs'],v['evidence']]))+' |')
 p=read(RUN/'results/accepted_pitch.json');lines+=['','## Pitch consistency','',f"Value: {p['value']}; uncodable: {p['uncodable']}; model-reported coverage complete: {p['coverage_complete']}.",p['evidence'],'', 'Limitations: '+ '; '.join(p['limitations'])]
 for pair in p['comparisons']:lines+=['',f"- t{pair['first']['turn_ref']} / t{pair['second']['turn_ref']}, counted={pair['counted']}: {pair['incompatibility']} Reconciliation: {pair['reconciliation']}"]
 calls=[json.loads(s) for s in (RUN/'calls.jsonl').read_text().splitlines()]
 usage={k:sum(c[k] for c in calls) for k in ['input_tokens','output_tokens','thinking_tokens']}
 lines+=['','## Run accounting','',f'Successful API generations: {len(calls)}. Usage: {usage}.','All prompts, raw outputs, structural rejection records and final accepted records are preserved. Accepted means structural checks passed, not human validation.']
 (RUN/'comparison.md').write_text('\n'.join(lines)+'\n')
 print(RUN/'comparison.md')
if __name__=='__main__':main()
