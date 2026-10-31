"""Read-only adapter for experimental Stage B JSON runs.

No registry writes, codebook regeneration, API clients or annotation calls. Statistics use
only validated accepted issue records; artifacts and rejected output remain inspectable.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from analysis.auc import grouped_auc,feature_value,mean

FEATURES={'response':'RS-RESP','evidence':'RS-EVID','substance':'PREP-SUBS','coherence':'PREP-COH','probe_severity':'PR-SEV','promotion_probe':'PR-FRAME'}
MODELS={'positivity_only':['positivity'],'interaction':['response','evidence','substance','coherence','repair','n_exchanges'],'plus_positivity':['positivity','response','evidence','substance','coherence','repair','n_exchanges'],'probe_control':['positivity','response','evidence','substance','coherence','repair','n_exchanges','probe_severity','promotion_probe'],'response_only':['response','evidence','substance','coherence','n_exchanges']}

def read_json(path,default=None):
    p=Path(path)
    return json.loads(p.read_text()) if p.exists() else default

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@dataclass
class RunData:
    root:Path
    frontend:Path|None
    loaded_at:str
    videos:pd.DataFrame
    ratings:pd.DataFrame
    exchanges:pd.DataFrame
    issues:pd.DataFrame
    decisions:pd.DataFrame
    calls:pd.DataFrame
    artifacts:pd.DataFrame
    flags:pd.DataFrame
    provenance:pd.DataFrame
    inputs:list[dict]

    def changes_since_load(self):
        changed=[]
        for x in self.inputs:
            p=Path(x['path'])
            if not p.exists() or (p.stat().st_mtime_ns,p.stat().st_size)!=(x['mtime_ns'],x['size']):changed.append(str(p))
        # New files also matter when annotation is still running.
        previous={x['path'] for x in self.inputs}
        for root in [self.root,self.frontend]:
            if root and root.exists():
                for pattern in ['st_*/completion.json','st_*/results/accepted_*.json']:
                    changed.extend(str(p) for p in root.glob(pattern) if str(p) not in previous)
        return sorted(set(changed))


def load_run(run,frontend=None,input_per_m=2.,output_per_m=12.):
    # These gate helpers are read-only. Statistical functions below do not depend on pipeline internals.
    from pipeline.stage_b.exchanges import check,validate_ratings
    root=Path(run).resolve();frontend=Path(frontend).resolve() if frontend else None
    selected=read_json(frontend/'selection.json',{}) if frontend else {}
    selection={r['video_id']:r for r in selected.get('videos',[])}
    existing={p.name for p in root.glob('st_*') if p.is_dir()}
    if not selection:selection={v:{} for v in existing}
    ids=sorted(set(selection)|existing)
    videos=[];ratings=[];exchanges=[];issues=[];decisions=[];calls=[];artifacts=[];flags=[];provenance=[];inputs={}
    def track(p):
        p=Path(p)
        if p.exists():
            stat=p.stat();inputs[str(p)]={'path':str(p),'mtime_ns':stat.st_mtime_ns,'size':stat.st_size}
        return p
    def get(p,default=None):return read_json(track(p),default)
    if frontend:track(frontend/'selection.json')
    def flag(vid,kind,path,message,active=True):flags.append(dict(video_id=vid,kind=kind,path=str(path),message=str(message),active=active))
    for vid in ids:
        r=root/vid;fr=frontend/vid if frontend else None
        meta=get(r/'experiment.json',{}) or (get(fr/'experiment.json',{}) if fr else {})
        completion=get(r/'completion.json',{})
        bounds=get(r/'boundaries.json',{}) or (get(fr/'boundaries.json',{}) if fr else {})
        prep=get(r/'results/accepted_prepared.json',{}) or (get(fr/'results/accepted_prepared.json',{}) if fr else {})
        state=completion.get('status','incomplete' if r.exists() else 'no_backend')
        title=Path(meta.get('source',selection.get(vid,{}).get('path',vid))).stem
        book_ok=True
        for name,expected in meta.get('codebook_sha256',{}).items():
            p=r/'codebooks'/name
            if not p.exists() and fr:p=fr/'codebooks'/name
            track(p);actual=sha(p) if p.exists() else None;ok=actual==expected;book_ok &=ok
            provenance.append(dict(video_id=vid,file=name,expected_sha256=expected,actual_sha256=actual,match=ok,path=str(p)))
            if not ok:flag(vid,'codebook_hash',p,'Archived codebook missing or changed')
        for stage,folder in [('backend',r),('frontend',fr)]:
            if folder is None or not folder.exists():continue
            cp=track(folder/'calls.jsonl')
            if cp.exists():
                for n,line in enumerate(cp.read_text().splitlines(),1):
                    if not line.strip():continue
                    try:c=json.loads(line)
                    except json.JSONDecodeError:
                        flag(vid,'partial_ledger',cp,f'Could not read line {n}');continue
                    inp=c.get('input_tokens');out=c.get('output_tokens');think=c.get('thinking_tokens')
                    cost=(inp*input_per_m+(out+think)*output_per_m)/1e6 if all(isinstance(v,(int,float)) for v in [inp,out,think]) else np.nan
                    calls.append(dict(video_id=vid,stage=stage,key=c['key'],phase=c.get('info',{}).get('phase','unknown'),model=c.get('model_version',c.get('model_requested')),input_tokens=inp,output_tokens=out,thinking_tokens=think,cost_usd=cost,seconds=c.get('seconds'),prompt_hash=c.get('prompt_hash'),ledger_path=str(cp),ledger_line=n,media=c.get('media'),selected=vid in selection))
            for area,pattern in [('prompts','*.txt'),('prompts','*.schema.json'),('raw','*.txt'),('raw','*_error.json'),('results','*.json'),('codebooks','*.md'),('templates','*')]:
                for p in sorted((folder/area).glob(pattern)):
                    if not p.is_file():continue
                    track(p);is_error=p.name.endswith('_error.json');key=p.stem.split('__attempt')[0]
                    kind='error' if is_error else area
                    artifacts.append(dict(video_id=vid,stage=stage,kind=kind,name=p.name,key=key,path=str(p),bytes=p.stat().st_size))
                    if is_error:
                        e=get(p,{});flag(vid,'historical_api_error',p,e.get('error_type','')+': '+e.get('message',''),False)
            for p in folder.glob('failure_*.json'):
                track(p);flag(vid,'historical_failure',p,get(p,{}).get('error',''),False)
        for p in (r/'results').glob('needs_human_review_*.json'):
            f=get(p,{});suffix=p.stem.removeprefix('needs_human_review_');active=not (r/f'results/accepted_{suffix}.json').exists()
            flag(vid,'review_required',p,f.get('errors',f.get('error',f)),active)
        if (r/'needs_human_review.json').exists():flag(vid,'map_review',r/'needs_human_review.json',get(r/'needs_human_review.json'),not (r/'frozen_bundle.json').exists())
        records={};eprecords={};mapped=[];map_valid=False;issue_total=0
        bundle=get(r/'frozen_bundle.json')
        if bundle:
            try:
                m,t=check(bundle);map_valid=True;mapped=m.exchanges;issue_total=sum(s.kind=='issue' for s in m.segments)
                if not book_ok:raise ValueError('Cannot accept ratings under mismatched codebook hashes')
                for s in m.segments:
                    if s.kind!='issue':continue
                    p=r/f'results/accepted_{s.segment_id}.json';a=get(p)
                    accepted=False;why='no accepted file'
                    if a:
                        try:
                            assert a['map_hash']==bundle['map_hash'] and a['timeline_hash']==bundle['timeline_hash'],'stale accepted hashes'
                            lo=min(z.start_s for z in t.turns if z.turn_idx in s.turn_refs);ctx=[z.turn_idx for z in t.turns if z.start_s<lo and z.end_s>max(0,lo-10)]
                            errs=validate_ratings(a,s,bundle,ctx);assert not errs,str(errs)
                            accepted=True;why='accepted';eprecords[s.segment_id]=a['episode'];records.update({x['exchange_id']:x['ratings'] for x in a['exchanges']})
                            for code,value in a['episode'].items():ratings.append(dict(video_id=vid,segment_id=s.segment_id,exchange_id=None,investor=None,unit='EPISODE',code=code,**value,artifact=str(p)))
                            for x in a['exchanges']:
                                actor=next(z.investor for z in mapped if z.exchange_id==x['exchange_id'])
                                for code,value in x['ratings'].items():ratings.append(dict(video_id=vid,segment_id=s.segment_id,exchange_id=x['exchange_id'],investor=actor,unit='EXCHANGE',code=code,**value,artifact=str(p)))
                        except (ValueError,KeyError,AssertionError) as exc:why=str(exc);flag(vid,'accepted_validation',p,why)
                    issues.append(dict(video_id=vid,segment_id=s.segment_id,topic=s.topic,accepted=accepted,status=why,cut=s.cut,turn_refs=s.turn_refs,start_s=min(z.start_s for z in t.turns if z.turn_idx in s.turn_refs),end_s=max(z.end_s for z in t.turns if z.turn_idx in s.turn_refs)))
                ts={z.turn_idx:z for z in t.turns}
                for x in mapped:
                    values=records.get(x.exchange_id,{})
                    exchanges.append(dict(video_id=vid,exchange_id=x.exchange_id,segment_id=x.segment_id,investor=x.investor,accepted=x.exchange_id in records,probe=x.probe,answer=x.answer,next_move=x.next_move,backchannels=x.backchannels,uptake_status=x.uptake_status,start_s=min(ts[i].start_s for i in x.probe),answered=bool(x.answer),next_same_investor=bool(x.next_move) and ts[x.next_move[0]].speaker==x.investor,eligibility=x.eligibility,**{k:feature_value(c,values[c]) if c in values else None for k,c in FEATURES.items()}))
            except (ValueError,KeyError,AssertionError) as exc:flag(vid,'map_validation',r/'frozen_bundle.json',exc);map_valid=False
        active=[x for x in flags if x['video_id']==vid and x['active']]
        finished=state in ('complete','complete_with_flags')
        analysis_eligible=finished and map_valid and book_ok and not any(x['kind'] in ('accepted_validation','map_validation') for x in active)
        for shark in bounds.get('outcomes_model_reviewed_not_gold',{}).get('sharks',[]):
            xs=[x for x in mapped if x.investor==shark['speaker'] and x.exchange_id in records];own={x.segment_id for x in xs}
            row=dict(video_id=vid,title=title,speaker=shark['speaker'],offer=None if shark['ever_offer'] is None else int(shark['ever_offer']),n_exchanges=len(xs),n_mapped_exchanges=sum(x.investor==shark['speaker'] for x in mapped),n_issues=len(own),positivity=prep.get('pitch_positivity',{}).get('value'),analysis_eligible=analysis_eligible,selected=vid in selection,has_flags=bool(active),status=state)
            row.update({k:mean([feature_value(c,records[x.exchange_id][c]) for x in xs]) for k,c in FEATURES.items()});row['repair']=mean([feature_value('RP-ACK',eprecords[s]['RP-ACK']) for s in sorted(own)])
            decisions.append(row)
        pitch=get(r/'results/accepted_pitch.json',{})
        videos.append(dict(video_id=vid,title=title,selected=vid in selection,status=state,backend_exists=r.exists(),frontend_exists=bool(fr and fr.exists()),map_valid=map_valid,analysis_eligible=analysis_eligible,active_flags=len(active),issues_mapped=issue_total,issues_accepted=len(eprecords),exchanges_mapped=len(mapped),exchanges_accepted=len(records),cutoff_s=bounds.get('cutoff_s'),positivity=prep.get('pitch_positivity',{}).get('value'),pitch_consistency=pitch.get('value'),pitch_status='accepted' if pitch else 'not accepted',source=meta.get('source'),roster=json.dumps(bundle.get('roster',[])) if bundle else '[]'))
    # Log costs for additional frontend attempts outside the current selection separately.
    if frontend:
        for folder in frontend.glob('st_*'):
            if folder.name in ids:continue
            cp=track(folder/'calls.jsonl')
            if cp.exists():
                for n,line in enumerate(cp.read_text().splitlines(),1):
                    try:c=json.loads(line)
                    except json.JSONDecodeError:continue
                    inp=c.get('input_tokens',0);out=c.get('output_tokens',0);think=c.get('thinking_tokens',0)
                    calls.append(dict(video_id=folder.name,stage='frontend',key=c['key'],phase=c.get('info',{}).get('phase','unknown'),model=c.get('model_version'),input_tokens=inp,output_tokens=out,thinking_tokens=think,cost_usd=(inp*input_per_m+(out+think)*output_per_m)/1e6,seconds=c.get('seconds'),prompt_hash=c.get('prompt_hash'),ledger_path=str(cp),ledger_line=n,media=c.get('media'),selected=False))
    return RunData(root,frontend,datetime.now(timezone.utc).isoformat(),*(pd.DataFrame(x) for x in [videos,ratings,exchanges,issues,decisions,calls,artifacts,flags,provenance]),list(inputs.values()))


def construct_summary(ratings):
    rows=[]
    for (unit,code),g in ratings.groupby(['unit','code']):
        values=g.loc[~g.uncodable,'value'];numeric=pd.to_numeric(values,errors='coerce');counts=values.map(str).value_counts()
        rows.append(dict(unit=unit,code=code,n=len(g),codable=len(values),missing=int(g.uncodable.sum()),missing_pct=100*g.uncodable.mean(),mean=numeric.mean(),sd=numeric.std(),distinct=values.map(str).nunique(),modal_pct=100*counts.iloc[0]/len(values) if len(values) else np.nan,distribution=counts.to_dict(),missing_reasons=g.loc[g.uncodable,'missing_reason'].value_counts().to_dict()))
    return pd.DataFrame(rows)


def evaluate(data,exclude_flagged=False):
    d=data.decisions
    cohort=d[d.analysis_eligible & d.selected & d.offer.notna()].copy()
    if exclude_flagged:cohort=cohort[~cohort.has_flags]
    if cohort.video_id.nunique()<2 or cohort.offer.nunique()<2:return pd.DataFrame(),{},cohort
    records=cohort.where(pd.notna(cohort),None).to_dict('records');details={};rows=[]
    for name,features in MODELS.items():
        result=grouped_auc(records,features);details[name]=result
        correct=pairs=0.
        for vid in cohort.video_id.unique():
            p=[x for x in result['predictions'] if x['video_id']==vid];pos=[x['prediction'] for x in p if x['offer']==1];neg=[x['prediction'] for x in p if x['offer']==0]
            for a in pos:
                for b in neg:correct+=(a>b)+.5*(a==b);pairs+=1
        rows.append(dict(model=name,videos=cohort.video_id.nunique(),sharks=len(cohort),offers=int(cohort.offer.sum()),pooled_oof_auc=result['pooled_oof_auc'],mean_within_video_auc=result['mean_within_video_auc'],within_video_pair_auc=correct/pairs if pairs else np.nan,within_video_pairs=int(pairs),evaluable_video_folds=sum(x['within_video_auc'] is not None for x in result['folds'])))
    return pd.DataFrame(rows),details,cohort


def artifact_text(data,video_id,kind='prompts',stage='backend',contains='',index=0):
    a=data.artifacts;chosen=a[(a.video_id==video_id)&(a.kind==kind)&(a.stage==stage)]
    if contains:chosen=chosen[chosen.name.str.contains(contains,regex=False)]
    chosen=chosen.sort_values('name')
    if chosen.empty:return 'No matching archived artifact.'
    row=chosen.iloc[index];return str(row.path)+'\n\n'+Path(row.path).read_text()
