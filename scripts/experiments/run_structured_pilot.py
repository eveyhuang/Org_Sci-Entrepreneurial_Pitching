#!/usr/bin/env python3
"""HISTORICAL ROUTE (v2 instrument, archived 2026-10-06). Not current execution.
Full, isolated five-video structured Stage A/B pilot. Resumable paid API phases."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,shutil,sys,traceback
from pathlib import Path
from typing import Literal
from dataclasses import replace
from types import SimpleNamespace
from pydantic import Field
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.run_structured_copy_keyboard import StructuredExperiment,read,dump
from pipeline.stage_b.structured import *
from pipeline.schemas.stage_b_timeline import TimelineRecord,normalised_turns,clock_to_seconds
from pipeline.schemas.stage_b import SegmentationRecord
from pipeline.schemas.stage_a_perceived_delivery import StageADeliveryRecord
from pipeline.codebook import load_codebook
from pipeline.prompts import render_prompt
from pipeline.media import clip
from pipeline.config import ModelConfig

RUN=ROOT/'runs/stage_b_structured_pilot5'
class DecisionEvent(Strict):
    kind:Literal['offer','joins_offer','out','withdraws_offer']
    source_s:float
    evidence:str
class SharkOutcome(Strict):
    speaker:str
    ever_offer:bool|None
    events:list[DecisionEvent]
    evidence:str
class OutcomeAudit(Strict):
    n_pitches:int
    prepared_start_s:float
    prepared_end_s:float
    first_decision_turn:int|None
    first_decision_start_s:float|None
    sharks:list[SharkOutcome]
    deal:bool|None
    limitations:list[str]

def snapshot(row):
    run=RUN/row['video_id']
    if (run/'experiment.json').exists():return run
    for name in ['codebooks','templates','prompts','results','raw','media']: (run/name).mkdir(parents=True,exist_ok=True)
    for name in ['segmentation','episode','pitch']:
        load_codebook(ROOT/f'docs/codebooks/archive/stage_b_v2_structured_2026-09-27/{name}.draft.yaml')
        for ext in ['md','yaml']:shutil.copy2(ROOT/f'docs/codebooks/archive/stage_b_v2_structured_2026-09-27/{name}.draft.{ext}',run/'codebooks')
    for ext in ['md','yaml']:shutil.copy2(ROOT/f'docs/codebooks/archive/stage_a_perceived_delivery_v2/stage_a_perceived_delivery.{ext}',run/'codebooks')
    TEMPLATE_SOURCES={'stage_b_timeline.j2':'docs/codebooks/archive/stage_b_v1_episode','stage_b_segmentation.j2':'docs/codebooks/archive/stage_b_v1_episode','stage_b_episode.j2':'docs/codebooks/archive/stage_b_v1_episode','stage_a_perceived_delivery.j2':'docs/codebooks/archive/stage_a_perceived_delivery_v2'}
    for name,src in TEMPLATE_SOURCES.items():shutil.copy2(ROOT/src/name,run/'templates')
    shutil.copy2(ROOT/'docs/codebooks/archive/stage_a_perceived_delivery_v2/stage_a_perceived_delivery_protocol.md',run/'templates')
    dump(run/'experiment.json',{'video_id':row['video_id'],'source':row['path'],'duration_s':row['duration_s'],'source_sha256':hashlib.sha256(Path(row['path']).read_bytes()).hexdigest(),'model':'gemini-3.1-pro-preview','temperature':0,'codebook_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'codebooks').iterdir()},'human_validated':False})
    return run

def namespace_prepared_turns(prepared,qa_ids):
    """Keep both source windows when a model reused full-timeline IDs in Q&A repair."""
    if not ({t['turn_idx'] for t in prepared}&set(qa_ids)):return prepared,{}
    base=max([1000000,*qa_ids])+1
    mapping={t['turn_idx']:base+i for i,t in enumerate(prepared)}
    return [{**t,'turn_idx':mapping[t['turn_idx']]} for t in prepared],mapping

class BudgetRecoveryProvider:
    def __init__(self,provider):self.provider=provider
    def upload_media(self,*args,**kwargs):return self.provider.upload_media(*args,**kwargs)
    def generate(self,request):
        if request.unit_key.endswith('_medium'):
            # Isolated per-video client proxy; preserve provider parsing and error handling.
            original=self.provider.client
            def generate_content(**kwargs):
                kwargs['config']={**kwargs['config'],'thinking_config':{'thinking_level':'medium'}}
                return original.models.generate_content(**kwargs)
            self.provider.client=SimpleNamespace(files=original.files,models=SimpleNamespace(generate_content=generate_content))
            try:return self.provider.generate(request)
            finally:self.provider.client=original
        if request.unit_key.endswith('_budget64'):
            request=replace(request,model=ModelConfig(request.model.name,request.model.temperature,65536))
        return self.provider.generate(request)

class Pilot(StructuredExperiment):
    def __init__(self,run):
        super().__init__(run);self.provider=BudgetRecoveryProvider(self.provider)
    def call(self,key,prompt,schema,media,info):
        """Recovery chain, all within one run: default -> (MAX_TOKENS) 64k output -> (deadline or read timeout)
        medium thinking. A default call that times out goes straight to medium thinking. Earlier runs' errors
        (raw/*_error.json) route a rerun directly to the step that has not failed yet."""
        base=super().call;res=self.run/'results';raw=self.run/'raw'
        timed_out=lambda m:'DEADLINE_EXCEEDED' in m or 'timed out' in m
        messages=lambda k:[read(p).get('message','') for p in raw.glob(f'{k}__attempt*_error.json')]
        medium=lambda:base(key+'_medium',prompt,schema,media,{**info,'recovery_thinking_level':'medium','recovery_max_output_tokens':32768})
        def budget64():
            try:return base(key+'_budget64',prompt,schema,media,{**info,'recovery_max_output_tokens':65536})
            except Exception as exc:
                if timed_out(str(exc)) or type(exc).__name__=='ReadTimeout':return medium()
                raise
        # results/<key>.json can be a copy of a recovery result (timeline.json for later readers); it then records
        # the recovery key and is not this call's own cache
        own=(res/f'{key}.json').exists() and read(res/f'{key}.json').get('key',key)==key
        if not own:   # cached results first, in chain order; then earlier runs' errors
            if (res/f'{key}_budget64.json').exists():return budget64()
            if (res/f'{key}_medium.json').exists() or any(timed_out(m) for m in messages(key)+messages(key+'_budget64')):return medium()
            if any('MAX_TOKENS' in m for m in messages(key)):return budget64()
        try:return base(key,prompt,schema,media,info)
        except Exception as exc:
            if 'MAX_TOKENS' in str(exc):return budget64()
            if timed_out(str(exc)) or type(exc).__name__=='ReadTimeout':return medium()
            raise
    def discovery(self):
        p=(self.run/'templates/stage_b_timeline.j2').read_text()
        tl=self.call('timeline',p,TimelineRecord,self.source,{'phase':'fresh_full_timeline'})
        res=self.run/'results'
        if not (res/'timeline.json').exists():  # a recovery key (_budget64/_medium) holds the record; readers expect timeline.json
            src=next(res/f'timeline{s}.json' for s in ('_budget64','_medium') if (res/f'timeline{s}.json').exists());shutil.copy2(src,res/'timeline.json')
        p=render_prompt('stage_b_segmentation.j2',codebook=self.segbook,context={'roster':json.dumps(tl['roster']),'timeline':json.dumps(tl['turns'])},prompt_dir=self.run/'templates')
        seg=self.call('segmentation',p,SegmentationRecord,self.source,{'phase':'draft_segmentation'})
        if seg['n_pitches']!=1:raise ValueError('Not a single pitch; review sample eligibility')
        p='Watch the ENTIRE supplied video to verify source boundaries and each investor\'s on-air offer outcome. Use this fixed roster; do not quote dialogue or use outside knowledge. '+json.dumps(tl['roster'])+'\nFind prepared_start_s (first founder presentation, exclude narrator/teaser), prepared_end_s (first substantive investor interaction). Find the START OF THE ENTIRE FIRST INVESTOR TURN containing an exit, offer or joining an offer, even if that turn begins with a question. Return first_decision_turn using the source timeline; verify source timing separately. A video cutoff or out is not a deal. For EVERY investor list all observable offer/join/out/withdrawal events, source seconds, concise evidence. ever_offer=true if they EVER make/join a concrete offer even if rejected or later withdrawn; false only if observed out with no preceding offer. No observed decision or ambiguous outcome=null. Nonbinding compliments/interest/valuation discussion are not offers. Return one shark per investor, no founders. Do not infer post-show closing. No expected outcomes or old annotations supplied.\nFULL TIMELINE\n'+json.dumps(tl['turns'])
        out=self.call('outcome_review',p,OutcomeAudit,self.source,{'phase':'outcome_and_cutoff_review'})
        investors={r['speaker'] for r in tl['roster'] if r['role']=='investor'}
        problem=self.outcome_problem(out,investors)
        if problem:  # one re-inspection with the violated rule named; a second failure stops the video
            out=self.call('outcome_review_retry',p+'\nA previous answer was rejected: '+problem+'. Re-inspect the source; apply the ever_offer rules exactly.',OutcomeAudit,self.source,{'phase':'outcome_and_cutoff_review','retry_reason':problem})
        problem=self.outcome_problem(out,investors)
        if problem:raise ValueError(problem)
        byid={t['turn_idx']:t for t in normalised_turns(tl)}
        if out['first_decision_turn'] is None or out['first_decision_start_s'] is None:raise ValueError('No observed decision cutoff; manual review needed')
        cutoff=min(out['first_decision_start_s'],byid[out['first_decision_turn']]['start_s'])
        if not 0<=out['prepared_start_s']<out['prepared_end_s']<cutoff:raise ValueError('Invalid prepared/decision boundaries')
        dump(self.run/'boundaries.json',{'prepared_start_s':out['prepared_start_s'],'prepared_end_s':out['prepared_end_s'],'cutoff_s':cutoff,'timeline_hash':digest(tl),'outcomes_model_reviewed_not_gold':out})

    @staticmethod
    def outcome_problem(out,investors):
        if len(out['sharks'])!=len(investors) or {s['speaker'] for s in out['sharks']}!=investors:return 'Outcome roster mismatch'
        for s in out['sharks']:
            if s['ever_offer'] is True and not any(e['kind'] in ['offer','joins_offer'] for e in s['events']):return f"Positive outcome lacks offer evidence ({s['speaker']})"
            if s['ever_offer'] is False and (any(e['kind'] in ['offer','joins_offer'] for e in s['events']) or not any(e['kind']=='out' for e in s['events'])):
                return f"Negative outcome invalid ({s['speaker']}: ever_offer=false requires an observed out and no offer; use null when no decision is observed)"
        return None

    def inputs(self):
        bounds=read(self.run/'boundaries.json');full=read(self.run/'results/timeline.json')['record'];roster=full['roster'];start=bounds['prepared_end_s'];end=bounds['cutoff_s']
        media=clip(self.source,start,end,self.run/'media')
        note=f'This Q&A clip begins at original-source second {start} and ends at {end}, before the first decision. Return ALL timestamps in ORIGINAL SOURCE seconds (clip seconds + {start}); do not use clip-relative seconds. Fixed roster: '+json.dumps(roster)
        return bounds,full,roster,start,end,media,note

    def timeline(self):
        """Repair and review the Q&A timeline against the source; gate it; write reviewed_timeline.json."""
        bounds,full,roster,start,end,media,note=self.inputs()
        seed=[t for t in normalised_turns(full) if t['end_s']>start and t['start_s']<end]
        p=note+'\nInspect the entire audiovisual clip and REPAIR this provisional timeline. Preserve all short spoken exchanges, acknowledgments, compliments/responses, overlapping speech and interrupted continuations. Remove non-speech laughter as a turn. Split explicit unrelated topic shifts that affect eligibility. Preserve IDs for retained events; added/split events use unused IDs >=10000. Endpoints must lie inside supplied clip. Action tags only, no dialogue transcription. Do not infer speech from a reaction shot. Every uncertain passage must be reviewed; use needs_repair for unresolved reparable defects, unobservable for genuinely unknowable source evidence. affected_codes MUST name code IDs (e.g. RAP-INTERRUPT, UP-TYPE), NEVER turn IDs.\nPROVISIONAL TURNS\n'+json.dumps(seed)
        proposal=self.call('timeline_repair',p,TimelineReview,media,{'phase':'timeline_repair','start':start,'end':end})
        p=note+'\nVerify the entire proposed timeline against audio/video; return the COMPLETE corrected timeline. Retain stable IDs, new IDs >=20000. Check every short exchange, speaker, interrupted continuation, overlap and within-speaker topic boundary. Do not simply approve. Mark material unresolved defects needs_repair, genuinely unavailable evidence unobservable with actual affected code IDs, not turn IDs. No quotations.\nPROPOSAL\n'+json.dumps(proposal)
        final=self.call('timeline_review',p,TimelineReview,media,{'phase':'timeline_review'})
        for attempt in range(3):
            tl=TimelineReview.model_validate(final)
            try:validate_timeline(tl,roster,start,end);break
            except ValueError as exc:
                dump(self.run/f'timeline_gate_rejection_{attempt}.json',{'error':str(exc),'record':final})
                if attempt==2:raise
                p=note+'\nRepair this timeline against the source. Gate rejected: '+str(exc)+'. Resolve reparable issues; do not label them unobservable merely to pass. Use genuine code IDs under affected_codes. Preserve IDs.\n'+json.dumps(final)
                final=self.call(f'timeline_gate_repair_{attempt+1}',p,TimelineReview,media,{'phase':'timeline_gate_repair'})
        dump(self.run/'reviewed_timeline.json',final)
        return bounds,full,roster,start,end,media,note,tl

    def frontend(self):
        """Everything the current Stage B back end (run_exchange_pilot.py) needs, and nothing it does not:
        full timeline, segmentation, outcomes and cutoff, Q&A timeline repair and review, Stage A v2."""
        self.discovery()
        bounds,*_=self.timeline()
        self.prepared(bounds)
        dump(self.run/'frontend_complete.json',{'status':'complete','human_validated':False})

    def structured(self):
        bounds,full,roster,start,end,media,note,tl=self.timeline()
        definitions='\n'.join(e.id+'\n'+e.definition+'\n'+e.include_if+'\n'+e.exclude_if for e in self.segbook.entries)
        rules='\nBuild issue episodes and explicit exchange spans. Assign every Q&A turn exactly once to an episode or exclusion. No target episode count. Exclude unrelated founder monologues after issue closure; preserve short social exchanges inside their issue. opening_probe=investor question, initial_answer=founder speech until substantive investor intervention OR unrelated topic shift. first_uptake=ONE earliest investor move after that answer, not a group of sequential moves. If first actor/order is inseparable use uncertain with empty uptake refs. minimal_backchannels includes ONLY listener tokens DURING INITIAL ANSWER that may be skipped, never later interjections. Later cycles have separate probe/answer/uptake links. Do not replace the initial answer with a later fuller answer. Cite stable IDs, never create them in the map. If a turn crosses an eligibility boundary, needs_repair is required. Preserve actual overlapping times; cannot silently change the timeline.\nCODEBOOK\n'+definitions+'\nTIMELINE\n'+tl.model_dump_json()
        proposal=self.call('interaction_map',note+rules,InteractionMap,media,{'phase':'interaction_map'})
        final=self.call('map_review',note+'\nVerify and correct the complete map against source. Do not approve by default.\n'+rules+'\nPROPOSED MAP\n'+json.dumps(proposal),InteractionMap,media,{'phase':'map_review'})
        for attempt in range(4):
            mapping=InteractionMap.model_validate(final)
            try:
                bundle=freeze_bundle(mapping,tl,roster,self.meta['source_sha256'],start,end,'separate Gemini audiovisual review + deterministic gate; not human validated');break
            except (ValueError,KeyError) as exc:
                dump(self.run/f'map_gate_rejection_{attempt}.json',{'error':str(exc),'record':final})
                if attempt==3:raise
                p=note+'\nSource-review correction required. Gate rejected: '+str(exc)+'. Return complete corrected map; do not hide source issues to pass.\n'+rules+'\nREJECTED MAP\n'+json.dumps(final)
                final=self.call(f'map_gate_repair_{attempt+1}',p,InteractionMap,media,{'phase':'map_gate_repair','attempt':attempt+1})
        dump(self.run/'frozen_bundle.json',bundle)

    def prepared(self,bounds):
        book=load_codebook(self.run/'codebooks/stage_a_perceived_delivery.yaml')
        p=render_prompt('stage_a_perceived_delivery.j2',codebook=book,prompt_dir=self.run/'templates')
        media=clip(self.source,bounds['prepared_start_s'],bounds['prepared_end_s'],self.run/'media')
        r=self.call('prepared',p,StageADeliveryRecord,media,{'phase':'stage_a_v2','start':bounds['prepared_start_s'],'end':bounds['prepared_end_s']})
        dump(self.run/'results/accepted_prepared.json',r)

    def episode(self,ep,bundle,tl):
        turns={t.turn_idx:t for t in tl.turns};start=min(turns[i].start_s for i in ep.turn_refs);end=max(turns[i].end_s for i in ep.turn_refs);media_start=max(0,start-10)
        context=[t for t in tl.turns if t.start_s<start and t.end_s>media_start]
        media=clip(self.source,media_start,end,self.run/'media')
        ctx={'roster':json.dumps(bundle['roster']),'pitch_summary':'Scored issue: '+ep.topic,'timeline_slice':'\n'.join(f't{i} SOURCE {turns[i].start_s}–{turns[i].end_s}; CLIP {max(0,turns[i].start_s-media_start):.2f}–{turns[i].end_s-media_start:.2f}; {turns[i].speaker}: {turns[i].tag}' for i in ep.turn_refs)}
        p=render_prompt('stage_b_episode.j2',codebook=self.epbook,context=ctx,prompt_dir=self.run/'templates')
        p+='\nBINDING REVIEWED WINDOWS\n'+ep.model_dump_json()+'\nALLOWED REFS PER CODE\n'+json.dumps({e.id:sorted(allowed_refs(e.id,ep)) for e in self.epbook.entries})
        p+=f'\nClip seconds 0–{start-media_start} are CONTEXT ONLY, eligible solely for RP-RECOV baseline. Episode scoring begins at clip {start-media_start}. Context turns: '+json.dumps([t.model_dump() for t in context])
        p+='\nCode the designated initial answer, never a later answer. Codable RS/PREP must cite initial answer; codable UP-TYPE must cite both answer and uptake. Codable negatives require inspected refs. Solo-founder TM codes require uncodable. Uncertain uptake requires uncodable, not NONE. Apply minimum-length/applicability before preparedness scores. If map contradicts video, mark affected code uncodable and request repair, do not silently choose another window. Issues: '+json.dumps(bundle['timeline']['issues']+bundle['mapping']['issues'])
        for attempt in range(1,4):
            key=f'episode_{ep.episode_id}_try{attempt}'
            record=self.call(key,p,self.epschema,media,{'phase':'episode','episode_id':ep.episode_id,'start':start,'end':end,'media_start':media_start,'timeline_hash':bundle['timeline_hash'],'map_hash':bundle['map_hash']})
            errors=validate_codes(record,ep,bundle,[t.turn_idx for t in context]);dump(self.run/f'results/{key}_qc.json',errors)
            if not errors:break
            p+='\nRejected structural output. Recode from source, not just relabel refs. Errors: '+json.dumps(errors)
        else:raise ValueError('Episode rejected after retries: '+ep.episode_id)
        source_key=next(k for k in [key,key+'_budget64',key+'_medium'] if (self.run/'results'/f'{k}.json').exists())
        accepted={'record':record,'source_call':source_key,'status':'structurally_accepted_not_human_validated','map_hash':bundle['map_hash'],'timeline_hash':bundle['timeline_hash']}
        if not record['UP-AFFECT']['uncodable']:
            audit_media=clip(self.source,start,end,self.run/'media')
            q=f'Inspect only visual observability of {ep.probing_speaker}, not warmth or other codes. Source offset={start}: return ORIGINAL SOURCE seconds, clip seconds + {start}. Roster: '+json.dumps(bundle['roster'])+'\nList all intervals of readable, identifiable, interaction-linked visibility of this investor. Exclude distant unreadable panel shots, ambiguous reaction cutaways, other people and offscreen speech. Do not aim for a coverage percentage. Each interval needs linkage evidence; empty list is allowed.'
            for j in range(2):
                audit=self.call(f'visibility_{ep.episode_id}_{j}',q,VisibilityAudit,audit_media,{'phase':'visibility','start':start,'end':end,'map_hash':bundle['map_hash']})
                if all(start<=v['start_s']<v['end_s']<=end for v in audit['intervals']):break
                q+='\nInvalid timestamps. Return original source seconds between '+str(start)+' and '+str(end)+'. Previous response: '+json.dumps(audit)
            passed=visibility_gate(VisibilityAudit.model_validate(audit),ep.probing_speaker,start,end)
            accepted['visibility_gate_passed']=passed
            if not passed:
                dump(self.run/f'results/{ep.episode_id}_before_visibility.json',record)
                record['UP-AFFECT']={'value':None,'uncodable':True,'evidence':'Source-based visual coverage audit fails codebook gate; see visibility artifact.','turn_refs':ep.turn_refs}
        dump(self.run/f'results/accepted_{ep.episode_id}.json',accepted)

    def pitch(self,bundle,bounds,full):
        tl=TimelineReview.model_validate(bundle['timeline']);mapping=InteractionMap.model_validate(bundle['mapping'])
        prepared=[t for t in normalised_turns(full) if t['start_s']>=bounds['prepared_start_s'] and t['end_s']<=bounds['prepared_end_s']]
        prepared,id_mapping=namespace_prepared_turns(prepared,{t.turn_idx for t in tl.turns})
        if id_mapping:dump(self.run/'prepared_id_namespace.json',{'reason':'Prevent prepared/Q&A ID collision without changing frozen Q&A IDs or source times','original_to_pitch_id':id_mapping})
        suffix='_prepared_ids_v2' if id_mapping else ''
        turns={t['turn_idx']:t for t in prepared};turns.update({t.turn_idx:t.model_dump() for t in tl.turns})
        units={t['turn_idx']:'prepared' for t in prepared}
        for e in mapping.episodes:units.update({i:e.episode_id for i in e.turn_refs})
        founders={r['speaker'] for r in bundle['roster'] if r['role']=='founder'};founder_refs={i for i,t in turns.items() if t['speaker'] in founders}
        p='Code cross-unit PREP-CONSIST from ONLY eligible prepared/episode founder claims. Do not use excluded gaps or outside knowledge. Original source timestamps.\nCODEBOOK\n'+json.dumps([e.model_dump(exclude={'provenance','examples'}) for e in self.pitchbook.entries])+'\nTIMELINE\n'+json.dumps(list(turns.values()))+'\nROSTER\n'+json.dumps(bundle['roster'])+'\nELIGIBLE TURN→UNIT\n'+json.dumps(units)+'\nEXCLUSIONS\n'+json.dumps([e.model_dump() for e in mapping.excluded])
        p+='\nFor each candidate pair give BOTH statements\' metric, period, units and scope and locatable founder refs/times. Count only established cross-unit incompatibility under matching scope. Same-unit differences, different periods, projections versus actuals and explicit corrections do not count. Missing scope affecting total requires uncodable. List rejected compatible candidates if comparable_claims_present=true; do not invent a pair if none exist. Report inspected_turn_refs for all eligible turns inspected, including investor context; coverage is model-reported, not human validation. All counted pairs must sum to value.'
        media=clip(self.source,0,bounds['cutoff_s'],self.run/'media')
        for attempt in range(1,4):
            r=self.call(f'pitch_try{attempt}{suffix}',p,PitchResult,media,{'phase':'pitch','eligible':sorted(units),'map_hash':bundle['map_hash'],'timeline_hash':bundle['timeline_hash']})
            errors=validate_pitch(PitchResult.model_validate(r),set(units),units,founder_refs,turns);dump(self.run/f'results/pitch_try{attempt}{suffix}_qc.json',errors)
            if not errors:dump(self.run/'results/accepted_pitch.json',r);return
            p+='\nGate rejected: '+json.dumps(errors)+' Reinspect and return complete corrected result.'
        raise ValueError('Pitch rejected after retries')

    def pitch_only(self):
        bounds,full,_,_,_,_,_=self.inputs();bundle=read(self.run/'frozen_bundle.json');assert_frozen(bundle);self.pitch(bundle,bounds,full)

    def annotate(self):
        bounds,full,_,_,_,_,_=self.inputs();bundle=read(self.run/'frozen_bundle.json');assert_frozen(bundle)
        tl=TimelineReview.model_validate(bundle['timeline']);mapping=InteractionMap.model_validate(bundle['mapping'])
        if len(mapping.episodes)>25:raise ValueError('More than 25 episodes requires scope review')
        self.prepared(bounds)
        failures=[]
        for ep in mapping.episodes:
            try:self.episode(ep,bundle,tl)
            except Exception as exc:failures.append({'episode':ep.episode_id,'error':str(exc)});dump(self.run/'episode_failures.json',failures)
        self.pitch(bundle,bounds,full)
        if failures:raise ValueError('Some episodes failed; inspect episode_failures.json')
        dump(self.run/'completion.json',{'status':'complete','human_validated':False,'episodes':len(mapping.episodes),'qa_turns':len(tl.turns)})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['frontend','discovery','structured','annotate','pitch_only','all'],required=True);ap.add_argument('--run-name',default='stage_b_structured_pilot5');ap.add_argument('--video-ids',nargs='*');ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    global RUN;RUN=ROOT/'runs'/a.run_name
    rows=read(RUN/'selection.json')['videos'];rows=[r for r in rows if not a.video_ids or r['video_id'] in a.video_ids]
    def work(row):
        exp=Pilot(snapshot(row))
        try:
            for phase in (['discovery','structured','annotate'] if a.phase=='all' else [a.phase]):getattr(exp,phase)()
            print('PHASE COMPLETE',row['video_id'],a.phase,flush=True);return True
        except Exception as exc:
            dump(exp.run/f'failure_{a.phase}.json',{'error':str(exc),'traceback':traceback.format_exc()});print('PHASE FAILED',row['video_id'],str(exc),flush=True);return False
        finally:exp.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:success=list(pool.map(work,rows))
    if not all(success):raise SystemExit(1)
if __name__=='__main__':main()
