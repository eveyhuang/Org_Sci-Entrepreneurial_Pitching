#!/usr/bin/env python3
"""Render observed results only; never fills unperformed API calls with zero/absence."""
import argparse,csv,json
from pathlib import Path

def read(p):return json.loads(p.read_text())
def display(v):
 if v is None:return 'NOT RUN'
 if v['uncodable']:return 'uncodable'
 return str(v['value']).lower() if isinstance(v['value'],bool) else str(v['value'])
def esc(s):return str(s).replace('|','\\|').replace('\n',' ')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args();run=a.run_dir.resolve()
 old=[read(run/f'baseline/episode_{i}.json') for i in range(4)]
 matched=[read(run/f'results/matched_episode_{i}.json') if (run/f'results/matched_episode_{i}.json').exists() else None for i in range(4)]
 if not any(matched):raise SystemExit('No new API episode results: comparison is not yet available.')
 codes=list(old[0]['record']);rows=[]
 lines=['# Copy Keyboard: revised-codebook real-API comparison','', 'This is a one-video development comparison, not an accuracy or reliability benchmark. Historical baseline and new results are single draws. The revised instrument changes evidence windows; a changed score is not automatically a correction.','', 'Matched arm: original spans/timeline, revised draft, experimental prompt whose generic binary/count wording respects code-specific scope. Rebuilt arm additionally uses a fresh timeline, segmentation and labeled pre-probe context. No original expected ratings or audit corrections were supplied to the model.','', '| Code | Episode 0 | Episode 1 | Episode 2 | Episode 3 |','|---|---|---|---|---|']
 if (run/'quality_gate.md').exists():
  lines[2:2]=['**Diagnostic outputs only: timeline/segmentation did not pass the freeze gate. The pitch count is an unaccepted model proposal.** See [quality gate](quality_gate.md) and [analytic assessment](assessment.md).','']
 changed=0;available=0
 for code in codes:
  cells=[]
  for i in range(4):
   before=old[i]['record'][code];after=matched[i]['record'][code] if matched[i] else None
   change=None if after is None else (before['value'],before['uncodable'])!=(after['value'],after['uncodable'])
   if after is not None:available+=1;changed+=int(change)
   cell=f'{display(before)} → {display(after)}';cells.append('**'+cell+'**' if change else cell)
   rows.append({'episode':i,'code':code,'old_value':display(before),'new_value':display(after),'changed':change,'old_evidence':before['evidence'],'new_evidence':after['evidence'] if after else '', 'old_turn_refs':json.dumps(before['turn_refs']),'new_turn_refs':json.dumps(after['turn_refs']) if after else ''})
  lines.append('| '+code+' | '+' | '.join(cells)+' |')
 lines+=['',f'Observed changes: {changed}/{available} available matched cells. This is a change count, not an error-reduction or accuracy statistic.','']
 with (run/'matched_comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 if (run/'results/segmentation.json').exists():
  seg=read(run/'results/segmentation.json')['record'];lines+=['**Fresh segmentation (proposed, not human-validated)**','', '| Episode | Start | End | Probe | Topic | Resolution | EP-CUT |','|---|---|---|---|---|---|---|']
  for i,e in enumerate(seg['episodes']):lines.append('| '+' | '.join(map(esc,[i,e['start'],e['end'],e['probing_speaker'],e['topic'],e['resolution'],e['cut']]))+' |')
  lines+=['',f"Prepared end: {seg['prepared_end']}. On-air deal: {seg['deal']}. Decisions are segmentation outcomes, not supplied to episode coding.",'']
  for i,e in enumerate(seg['episodes']):
   p=run/f'results/rebuilt_episode_{i}.json'
   if not p.exists():lines += [f'Rebuilt episode {i}: NOT RUN.',''];continue
   r=read(p);info=r['info'];lines += [f"**Rebuilt episode {i}: {e['topic']} ({info['source_start_s']}–{info['source_end_s']} s)**",'', '| Code | Value | Turns | Evidence |','|---|---|---|---|']
   for code,v in r['record'].items():lines.append('| '+' | '.join(map(esc,[code,display(v),v['turn_refs'],v['evidence']]))+' |')
   lines+=['']
 if (run/'results/pitch.json').exists():
  r=read(run/'results/pitch.json')['record'];lines+=['**Pitch consistency: new result, no baseline comparator**','',f"PREP-CONSIST: {display(r['record']['PREP-CONSIST'])}.",r['record']['PREP-CONSIST']['evidence'],'',f"Coverage complete: {r['coverage_complete']}; comparable claims present: {r['comparable_claims_present']}."]
  for pair in r['comparisons']:lines.append(f"- t{pair['first_turn']} / t{pair['second_turn']}, counted={pair['counted']}: {pair['reason']}")
  lines+=['', 'Limitations: '+ '; '.join(r['limitations']),'']
 calls=[json.loads(s) for s in (run/'calls.jsonl').read_text().splitlines()] if (run/'calls.jsonl').exists() else []
 lines+=['**API usage for successful calls**','',f"Calls: {len(calls)}; input tokens: {sum(x['input_tokens'] for x in calls):,}; output tokens: {sum(x['output_tokens'] for x in calls):,}; thinking tokens: {sum(x['thinking_tokens'] for x in calls):,}.",'These totals exclude unreported provider usage from failed attempts; failures are retained under raw/. No dollar cost is inferred without a verified pricing basis.','', '[Machine-readable matched comparison](matched_comparison.csv). See experiment.json, calls.jsonl, prompts/, results/ and raw/ for frozen inputs and returned evidence.']
 (run/'comparison.md').write_text('\n'.join(lines)+'\n');print(run/'comparison.md')
if __name__=='__main__':main()
