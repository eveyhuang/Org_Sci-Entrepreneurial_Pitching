#!/usr/bin/env python3
"""Map v3 + pruned codebook vs the exchange pilot, on identical timelines. Read-only; no API calls.

Both runs start from the same reviewed timelines (the v3 run is seeded with --seed-timelines-from),
so turn IDs are shared and maps and ratings can be matched exchange by exchange.
    .venv/bin/python scripts/experiments/compare_map_v3.py [--old stage_b_exchange_pilot5] [--new stage_b_exchange_pilot5_v3]
Writes <new>/decision_features.csv (same per-shark features as compare_exchange_pilot.py) and
<new>/map_v3_comparison.json.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.experiments.analyze_structured_pilot import mean, feature_value
from pipeline.stage_b.exchanges import check

FEATURES = {'response': 'RS-RESP', 'evidence': 'RS-EVID', 'substance': 'PREP-SUBS', 'coherence': 'PREP-COH',
            'probe_severity': 'PR-SEV', 'promotion_probe': 'PR-FRAME'}
PRICE = dict(inp=2e-6, out=12e-6)  # gemini-3.1-pro-preview, USD per token (thinking billed as output)

def read(p): return json.loads(Path(p).read_text())

def load(run, vid):
    r = ROOT / 'runs' / run / vid
    b = read(r / 'frozen_bundle.json'); m, t = check(b)
    records, issues = {}, {}
    for s in m.segments:
        if s.kind != 'issue': continue
        f = r / f'results/accepted_{s.segment_id}.json'
        if not f.exists(): continue   # issue flagged for human review: no accepted ratings
        a = read(f)
        records.update({x['exchange_id']: x['ratings'] for x in a['exchanges']}); issues[s.segment_id] = a['episode']
    calls = [json.loads(l) for l in (r / 'calls.jsonl').read_text().splitlines()] if (r / 'calls.jsonl').exists() else []
    return r, b, m, t, records, issues, calls

def value(v): return None if v['uncodable'] else v['value']

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--old', default='stage_b_exchange_pilot5'); ap.add_argument('--new', default='stage_b_exchange_pilot5_v3'); a = ap.parse_args()
    videos = [p.name for p in sorted((ROOT / 'runs' / a.new).glob('st_*')) if (p / 'frozen_bundle.json').exists()]
    out = {'videos': {}, 'code_agreement': {}, 'issue_code_agreement': {}, 'cost': {}}
    code_pairs, issue_pairs, rows = defaultdict(list), defaultdict(list), []
    phase_cost = {a.old: defaultdict(float), a.new: defaultdict(float)}
    for vid in videos:
        ro, bo, mo, to, reco, isso, co = load(a.old, vid)
        rn, bn, mn, tn, recn, issn, cn = load(a.new, vid)
        same_timeline = bo['timeline_hash'] == bn['timeline_hash']
        fo = {x.turn_ref: x.kind for x in mo.investor_moves}; fn = {x.turn_ref: x.kind for x in mn.investor_moves}
        listed = {i for x in mo.exchanges for i in x.backchannels}   # v2 had no 'backchannel' label
        fo_eq = {i: ('backchannel' if i in listed else k) for i, k in fo.items()}
        coarse = lambda k: 'receipt' if k in ('receipt', 'backchannel') else k
        shared = set(fo) & set(fn)
        xo = {tuple(sorted(x.probe)): x for x in mo.exchanges}; xn = {tuple(sorted(x.probe)): x for x in mn.exchanges}
        both = {k for k in set(xo) & set(xn) if xo[k].exchange_id in reco and xn[k].exchange_id in recn}
        same_window = sum(sorted(xo[k].answer) == sorted(xn[k].answer) and xo[k].next_move == xn[k].next_move for k in both)
        for k in both:
            for code, v in recn[xn[k].exchange_id].items():
                if code in reco[xo[k].exchange_id]: code_pairs[code].append((value(reco[xo[k].exchange_id][code]), value(v)))
        so = {s.segment_id: set(s.turn_refs) for s in mo.segments if s.kind == 'issue'}
        sn = {s.segment_id: set(s.turn_refs) for s in mn.segments if s.kind == 'issue'}
        matched_issues = 0
        for i, A in sn.items():
            best = max(so, key=lambda j: len(A & so[j]) / len(A | so[j]), default=None)
            if best and i in issn and best in isso and len(A & so[best]) / len(A | so[best]) >= 0.5:
                matched_issues += 1
                for code, v in issn[i].items():
                    if code in isso[best]: issue_pairs[code].append((value(isso[best][code]), value(v)))
        out['videos'][vid] = dict(
            same_timeline=same_timeline,
            issues_old=len(so), issues_new=len(sn), issues_matched=matched_issues,
            exchanges_old=len(xo), exchanges_new=len(xn), same_probe=len(both), same_answer_and_next_move=same_window,
            functions_exact=sum(fo_eq[i] == fn[i] for i in shared), functions_coarse=sum(coarse(fo[i]) == coarse(fn[i]) for i in shared), investor_turns=len(shared),
            needs_human_review=sorted(p.stem.removeprefix('needs_human_review_') for p in (rn / 'results').glob('needs_human_review_*.json') if not (rn / 'results' / ('accepted_' + p.stem.removeprefix('needs_human_review_') + '.json')).exists()),
            map_calls_new=Counter(c['info'].get('phase') for c in cn if c['info'].get('phase', '').startswith('map')))
        for run, calls in ((a.old, co), (a.new, cn)):
            for c in calls: phase_cost[run][c['info'].get('phase')] += c['input_tokens'] * PRICE['inp'] + (c['output_tokens'] + c['thinking_tokens']) * PRICE['out']
        # per-shark features, identical formulas to compare_exchange_pilot.decision_rows(only_openers=False)
        bounds = read(rn / 'boundaries.json'); ts = {x.turn_idx: x for x in tn.turns}
        positivity = read(rn / 'results/accepted_prepared.json')['pitch_positivity']['value']
        for shark in bounds['outcomes_model_reviewed_not_gold']['sharks']:
            xs = [x for x in mn.exchanges if x.investor == shark['speaker'] and x.exchange_id in recn]; own = {x.segment_id for x in xs}
            row = {'video_id': vid, 'speaker': shark['speaker'], 'offer': None if shark['ever_offer'] is None else int(shark['ever_offer']),
                   'n_exchanges': len(xs), 'n_issues': len(own), 'positivity': positivity}
            for name, code in FEATURES.items(): row[name] = mean([feature_value(code, recn[x.exchange_id][code]) for x in xs])
            row['repair'] = mean([feature_value('RP-ACK', issn[i]['RP-ACK']) for i in sorted(own)])
            rows.append(row)
    for code, pairs in sorted(code_pairs.items()):
        both_coded = [(o, n) for o, n in pairs if o is not None and n is not None]
        out['code_agreement'][code] = dict(n=len(pairs), both_coded=len(both_coded), agree=round(sum(o == n for o, n in both_coded) / len(both_coded), 2) if both_coded else None,
                                           uncodable_old=sum(o is None for o, _ in pairs), uncodable_new=sum(n is None for _, n in pairs))
    for code, pairs in sorted(issue_pairs.items()):
        both_coded = [(o, n) for o, n in pairs if o is not None and n is not None]
        out['issue_code_agreement'][code] = dict(n=len(pairs), agree=round(sum(o == n for o, n in both_coded) / len(both_coded), 2) if both_coded else None)
    for run, pc in phase_cost.items(): out['cost'][run] = {k: round(v, 2) for k, v in sorted(pc.items(), key=lambda kv: -kv[1])} | {'TOTAL': round(sum(pc.values()), 2)}
    new = ROOT / 'runs' / a.new
    (new / 'map_v3_comparison.json').write_text(json.dumps(out, indent=2, default=dict) + '\n')
    with open(new / 'decision_features.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(out, indent=1, default=dict))

if __name__ == '__main__':
    main()
