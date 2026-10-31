"""V2 issue membership, actor-specific exchanges and local visual observations.

V1 remains readable in structured.py. Context links never confer episode membership.
"""
from __future__ import annotations
from typing import Literal
from pydantic import Field,create_model
from pipeline.stage_b.structured import Strict,TimelineReview,digest,validate_timeline
from pipeline.schemas.generate import build_record_model

EXCHANGE_CODES={'PR-FRAME','PR-SEV','RS-RESP','RS-FRAME','RS-EVID','RS-UNCERT','PREP-SUBS','PREP-COH','PREP-CTX','UP-TYPE'}
REASONS=Literal['none','no_opportunity','short_answer','not_applicable','off_camera','edited_adjacency','ambiguous_attribution','cutoff','inaudible','pipeline_defect']
class Segment(Strict):
    segment_id:str
    kind:Literal['issue','procedural','social','other']
    topic:str
    turn_refs:list[int]
    rationale:str
    cut:bool
class Move(Strict):
    turn_ref:int
    # backchannel = minimal listener token that does not take the floor ('mm-hm', 'right'); receipt = a reaction that does.
    kind:Literal['probe','receipt','backchannel','procedural','social','other']
    reason:str
class Exchange(Strict):
    exchange_id:str
    segment_id:str
    investor:str
    probe:list[int]
    answer:list[int]
    next_move:list[int]
    backchannels:list[int]
    uptake_status:Literal['observed','cutoff','edited_adjacency','ambiguous_attribution','no_opportunity']
    eligibility:str
    uptake_reason:str
class ExchangeCorrections(Strict):
    exchanges:list[Exchange]
    source_defects:list[str]
    review_notes:str

class Map(Strict):
    segments:list[Segment]
    exchanges:list[Exchange]
    investor_moves:list[Move]
    repair_requests:list[str]
    review_notes:str

class UptakeFlag(Strict):
    """A judgment the timeline cannot supply: the next investor move exists but its link to this answer is unobservable."""
    probe_turn:int
    status:Literal['edited_adjacency','ambiguous_attribution']
    reason:str
class MapProposal(Strict):
    """What the model decides. Exchanges (answer spans, listener tokens, next moves, uptake status) are derived by code."""
    segments:list[Segment]
    investor_moves:list[Move]
    uptake_flags:list[UptakeFlag]
    # Probe turns the model judges to be ONE question although the default rule would split them
    # (e.g. a reiteration after a minimal founder reply). Each group: one investor, one issue.
    probe_groups:list[list[int]]
    repair_requests:list[str]
    review_notes:str

def _timeline_index(timeline,roster):
    ts={t.turn_idx:t for t in timeline.turns};roles={r['speaker']:r['role'] for r in roster}
    order=sorted(ts,key=lambda i:(ts[i].start_s,ts[i].end_s,i))
    return ts,roles,order

def fill_unassigned(segments,timeline,moves,roster=()):
    """Put turns the map left out of every segment into 'other' segments, one per contiguous run.
    A run containing a probe is left unassigned so the gate still rejects it: a probe must belong to an issue.
    So is a run containing a founder turn that falls inside an issue's span (between its first and last
    turn): those are almost always answers the map failed to assign (2026-09-28, st_f601a0ddf8)."""
    ts={t.turn_idx:t for t in timeline.turns};order=sorted(ts,key=lambda i:(ts[i].start_s,ts[i].end_s,i))
    roles={r['speaker']:r['role'] for r in roster}
    spans=[(min(ts[i].start_s for i in s.turn_refs if i in ts),max(ts[i].start_s for i in s.turn_refs if i in ts)) for s in segments if s.kind=='issue' and any(i in ts for i in s.turn_refs)]
    inside_issue=lambda i:roles.get(ts[i].speaker)=='founder' and any(a<ts[i].start_s<z for a,z in spans)
    owned={i for s in segments for i in s.turn_refs};kinds={m.turn_ref:m.kind for m in moves}
    runs=[];current=[]
    for i in order:
        if i in owned:
            if current:runs.append(current);current=[]
        else:current.append(i)
    if current:runs.append(current)
    extra=[Segment(segment_id=f'auto_other_{n}',kind='other',topic='unassigned by map',turn_refs=run,rationale='Turns the map did not assign to any segment; kept outside every issue automatically.',cut=False)
           for n,run in enumerate(runs,1) if not any(kinds.get(i)=='probe' or inside_issue(i) for i in run)]
    return list(segments)+extra

def derive_exchanges(segments,moves,timeline,roster,uptake_flags=(),probe_groups=()):
    """Build every exchange from the model's judgments: turn functions, issue membership, explicit
    probe groupings and uptake flags. The rules match validate_map, so a derived map cannot fail
    it mechanically:
    - probe: consecutive probe turns by one investor in one issue with no founder turn between them,
      unless the model grouped probe turns explicitly;
    - answer: the issue's founder turns after the probe's first turn. Before the founder starts, only a new
      question pre-empts the answer (a side remark does not). Once the answer has started, any
      investor turn other than a minimal backchannel ends it, as does a founder turn in another segment;
    - backchannels: minimal listener tokens inside the answer; they never count as the next move;
    - next move: the earliest later investor turn that is not part of the probe and not a
      backchannel, anywhere before the cutoff. Status: observed / cutoff / no_opportunity, unless the
      model flagged the link as edited or ambiguously attributed.
    """
    ts,roles,order=_timeline_index(timeline,roster)
    owner={i:s.segment_id for s in segments for i in s.turn_refs};kinds={m.turn_ref:m.kind for m in moves};why={m.turn_ref:m.reason for m in moves}
    flags={f.probe_turn:f for f in uptake_flags}
    investor_refs={i for i in ts if roles.get(ts[i].speaker)=='investor'}
    explicit={i:tuple(g) for g in probe_groups for i in g}
    for g in probe_groups:
        if len({ts[i].speaker for i in g})!=1 or len({owner.get(i) for i in g})!=1 or any(kinds.get(i)!='probe' for i in g):
            raise ValueError(f'probe group {g}: must be probe turns by one investor in one issue')
    groups=[];open_group=None;seen=set()
    for i in order:
        role=roles.get(ts[i].speaker)
        if role=='founder':open_group=None;continue
        if kinds.get(i)!='probe' or i not in owner or i in seen:continue
        if i in explicit:
            members=sorted(explicit[i],key=order.index);groups.append({'investor':ts[i].speaker,'segment':owner[i],'probe':members});seen.update(members);open_group=None;continue
        g=open_group
        if g and g['investor']==ts[i].speaker and g['segment']==owner[i]:g['probe'].append(i)
        else:open_group={'investor':ts[i].speaker,'segment':owner[i],'probe':[i]};groups.append(open_group)
        seen.add(i)
    exchanges=[]
    for n,g in enumerate(groups,1):
        probe=g['probe'];seg=g['segment'];answer=[];listening=[]
        for j in order[order.index(probe[0])+1:]:  # from the first probe turn: a founder reply between grouped probe turns is part of the answer
            role=roles.get(ts[j].speaker)
            if role=='founder':
                if owner.get(j)!=seg:break
                answer.append(j);continue
            if role!='investor' or j in probe:continue
            if kinds.get(j)=='backchannel':
                if answer and owner.get(j)==seg:listening.append(j)
                continue
            if not answer and kinds.get(j)!='probe':continue
            break
        flagged=next((flags[i] for i in probe if i in flags),None)
        backchannels=[]
        if not answer:status,next_move='no_opportunity',[]
        else:
            a0=min(ts[i].start_s for i in answer);a1=max(ts[i].end_s for i in answer)
            backchannels=[j for j in listening if a0<=ts[j].start_s<a1]
            candidates=sorted((i for i in investor_refs if ts[i].start_s>a0 and i not in probe+backchannels and kinds.get(i)!='backchannel'),key=lambda i:(ts[i].start_s,ts[i].end_s,i))
            if flagged:status,next_move=flagged.status,[]
            elif candidates:status,next_move='observed',[candidates[0]]
            else:status,next_move='cutoff',[]
        reason=flagged.reason if flagged and answer else {'observed':'Derived: earliest later investor turn.','cutoff':'Derived: no later investor turn before the cutoff.','no_opportunity':'Derived: no founder answer before the next question.'}.get(status,'')
        exchanges.append(Exchange(exchange_id=f'x{n}',segment_id=seg,investor=g['investor'],probe=probe,answer=answer,next_move=next_move,
                                  backchannels=backchannels,uptake_status=status,eligibility='; '.join(why.get(i,'') for i in probe).strip('; ') or 'substantive probe',uptake_reason=reason))
    return exchanges

def derive_map(proposal,timeline,roster):
    """Turn a MapProposal into the full Map the gate, freeze and coding steps already use."""
    segments=fill_unassigned(proposal.segments,timeline,proposal.investor_moves,roster)
    return Map(segments=segments,exchanges=derive_exchanges(segments,proposal.investor_moves,timeline,roster,proposal.uptake_flags,proposal.probe_groups),
               investor_moves=proposal.investor_moves,repair_requests=proposal.repair_requests,review_notes=proposal.review_notes)

class MapCorrections(Strict):
    """A review's output: only what should change, not a rewritten map."""
    move_corrections:list[Move]          # investor turns whose function label changes
    segment_replacements:list[Segment]   # complete corrected segments (matched by segment_id) or new segments
    removed_segments:list[str]
    probe_groups:list[list[int]]         # the complete list after review (usually empty)
    uptake_flags:list[UptakeFlag]        # the complete list after review (usually empty)
    repair_requests:list[str]
    review_notes:str

def apply_corrections(proposal,corrections):
    """Apply a review's corrections to a proposal. Unknown segment IDs are added; removed IDs must exist.
    A correction for a turn the proposal did not label adds that label (the proposal omitted an investor
    turn); derive_map/validate_map still reject labels on turns that are not investor turns."""
    moves={m.turn_ref:m for m in proposal.investor_moves}
    for m in corrections.move_corrections:moves[m.turn_ref]=m
    segments={s.segment_id:s for s in proposal.segments}
    for sid in corrections.removed_segments:
        if sid not in segments:raise ValueError(f'cannot remove unknown segment {sid}')
        del segments[sid]
    for s in corrections.segment_replacements:segments[s.segment_id]=s
    return MapProposal(segments=list(segments.values()),investor_moves=sorted(moves.values(),key=lambda m:m.turn_ref),
                       uptake_flags=corrections.uptake_flags,probe_groups=corrections.probe_groups,
                       repair_requests=corrections.repair_requests,review_notes=(proposal.review_notes+'; review: '+corrections.review_notes).strip('; '))

def validate_map(mapping,timeline,roster,start,end):
    validate_timeline(timeline,roster,start,end)
    if mapping.repair_requests:raise ValueError('source timeline/map needs repair: '+str(mapping.repair_requests))
    ts={t.turn_idx:t for t in timeline.turns};roles={r['speaker']:r['role'] for r in roster};owners={};segments={}
    for s in mapping.segments:
        if s.segment_id in segments:raise ValueError('duplicate segment ID')
        segments[s.segment_id]=s
        if not s.turn_refs or len(s.turn_refs)!=len(set(s.turn_refs)):raise ValueError('empty/duplicate segment refs')
        for i in s.turn_refs:
            if i not in ts or i in owners:raise ValueError('unknown/shared membership')
            owners[i]=s.segment_id
    if set(owners)!=set(ts):raise ValueError('incomplete timeline membership; missing turn IDs '+str(sorted(set(ts)-set(owners))))
    moves={m.turn_ref:m for m in mapping.investor_moves}
    investor_refs={i for i,t in ts.items() if roles[t.speaker]=='investor'}
    if len(moves)!=len(mapping.investor_moves) or set(moves)!=investor_refs:raise ValueError('each investor turn needs one function label')
    probes=set();exids=set()
    exchange_errors=[]
    for x in mapping.exchanges:
        try:
            if x.exchange_id in exids:raise ValueError(x.exchange_id+': duplicate exchange ID')
            exids.add(x.exchange_id)
            if x.segment_id not in segments or segments[x.segment_id].kind!='issue':raise ValueError(x.exchange_id+': exchange must belong to issue')
            if not x.probe or not x.eligibility.strip():raise ValueError(x.exchange_id+': missing substantive probe justification')
            if len(x.probe)!=len(set(x.probe)) or probes&set(x.probe):raise ValueError(x.exchange_id+': duplicate probe ownership')
            probes.update(x.probe)
            if not set(x.probe+x.answer+x.next_move+x.backchannels)<=ts.keys():raise ValueError(x.exchange_id+': unknown exchange refs')
            if not set(x.probe+x.answer+x.backchannels)<=set(segments[x.segment_id].turn_refs):raise ValueError(x.exchange_id+': probe/answer/backchannel outside membership')
            if any(ts[i].speaker!=x.investor or moves.get(i) is None or moves[i].kind!='probe' for i in x.probe):raise ValueError(x.exchange_id+': probe actor/function mismatch')
            if roles.get(x.investor)!='investor' or any(roles[ts[i].speaker]!='founder' for i in x.answer):raise ValueError(x.exchange_id+': incorrect exchange roles')
            if (x.uptake_status=='observed')!=bool(x.next_move):raise ValueError(x.exchange_id+': next-move status mismatch')
            if not x.answer:
                if x.uptake_status!='no_opportunity':raise ValueError(x.exchange_id+': no answer must identify no opportunity')
                continue
            if min(ts[i].start_s for i in x.answer)<min(ts[i].start_s for i in x.probe):raise ValueError(x.exchange_id+': answer precedes probe')
            a0=min(ts[i].start_s for i in x.answer);a1=max(ts[i].end_s for i in x.answer)
            if any(roles[ts[i].speaker]!='investor' or not a0<=ts[i].start_s<a1 for i in x.backchannels):raise ValueError(x.exchange_id+': invalid skipped listener token')
            candidates=[i for i in investor_refs if ts[i].start_s>a0 and i not in x.probe+x.backchannels and moves[i].kind!='backchannel']
            candidates.sort(key=lambda i:(ts[i].start_s,ts[i].end_s,i))
            if x.next_move:
                if len({ts[i].speaker for i in x.next_move})!=1 or any(i not in investor_refs for i in x.next_move):raise ValueError(x.exchange_id+': next move must identify one investor')
                if not candidates or min(ts[i].start_s for i in x.next_move)!=ts[candidates[0]].start_s:raise ValueError(x.exchange_id+': next move '+str(x.next_move)+' skips earliest investor '+str(candidates[:1]))
                if max(ts[i].start_s for i in x.answer)>min(ts[i].start_s for i in x.next_move):raise ValueError(x.exchange_id+': answer crosses investor intervention')
            elif x.uptake_status=='cutoff' and candidates:raise ValueError(x.exchange_id+': observable investor exists before cutoff')
            if set(x.probe)&set(x.answer) or set(x.answer)&set(x.next_move):raise ValueError(x.exchange_id+': exchange component collision')
        except ValueError as exc:exchange_errors.append(str(exc))
    if exchange_errors:raise ValueError('; '.join(exchange_errors))
    if probes!={i for i,m in moves.items() if m.kind=='probe'}:raise ValueError('probe/exchange mismatch: uncovered probes '+str(sorted({i for i,m in moves.items() if m.kind=='probe'}-probes))+'; incorrectly linked probes '+str(sorted(probes-{i for i,m in moves.items() if m.kind=='probe'})))
    for s in mapping.segments:
        xs=[x for x in mapping.exchanges if x.segment_id==s.segment_id]
        if (s.kind=='issue')!=bool(xs):raise ValueError('issue requires substantive exchange; nonissue cannot own one')
        if xs:
            first=min(s.turn_refs,key=lambda i:ts[i].start_s)
            if first not in {i for x in xs for i in x.probe}:raise ValueError('issue begins before substantive probe')

def freeze(mapping,timeline,roster,source_sha,start,end):
    validate_map(mapping,timeline,roster,start,end)
    return dict(schema_version=2,mapping=mapping.model_dump(),timeline=timeline.model_dump(),roster=roster,source_sha256=source_sha,start_s=start,cutoff_s=end,map_hash=digest(mapping),timeline_hash=digest(timeline),human_validated=False,status='frozen_for_development')
def check(bundle):
    m=Map.model_validate(bundle['mapping']);t=TimelineReview.model_validate(bundle['timeline'])
    if digest(m)!=bundle['map_hash'] or digest(t)!=bundle['timeline_hash']:raise ValueError('stale V2 freeze')
    validate_map(m,t,bundle['roster'],bundle['start_s'],bundle['cutoff_s']);return m,t

def record_schema(book,unit,code_refs=None):
    base=build_record_model(book,name=unit+'Record',unit=unit);fields={}
    for name,f in base.model_fields.items():
        ref_type=Literal[tuple(str(i) for i in sorted(code_refs[f.alias]))] if code_refs else int
        rating=create_model(unit+'_'+name,__base__=f.annotation,
            missing_reason=(REASONS,...),
            turn_refs=(list[ref_type],Field(min_length=1,description='Inspected eligible refs, including for negative or missing ratings.')))
        fields[name]=(rating,Field(alias=f.alias))
    return create_model(unit+'WithReasons',__base__=Strict,**fields)
def batch_schema(book,include_episode=True,episode_refs=None,exchange_refs=None,exchanges=None):
    ex=record_schema(book,'EXCHANGE',exchange_refs);ep=record_schema(book,'EPISODE',episode_refs)
    item=create_model('ExchangeRatings',__base__=Strict,exchange_id=(str,...),ratings=(ex,...))
    wire=list[item]
    if exchanges and len(exchanges)>1:
        fields={}
        for j,x in enumerate(exchanges):
            scoped=record_schema(book,'EXCHANGE',{code:allowed(code,x) for code in EXCHANGE_CODES})
            fields['item_'+str(j)]=(scoped,Field(alias=x.exchange_id))
        wire=create_model('FixedExchangeRatings',__base__=Strict,**fields)
    fields={'exchanges':(wire,...)}
    if include_episode:fields['episode']=(ep,...)
    return create_model('IssueRatings',__base__=Strict,**fields)

def normalize_rating_refs(record):
    """Gemini enums must be strings; losslessly restore numeric stable IDs after parsing."""
    import copy
    out=copy.deepcopy(record)
    out.setdefault('episode',None)
    if isinstance(out['exchanges'],dict):out['exchanges']=[{'exchange_id':k,'ratings':v} for k,v in out['exchanges'].items()]
    groups=[out.get('episode') or {}]+[x['ratings'] for x in out['exchanges']]
    for group in groups:
        for value in group.values():value['turn_refs']=[int(i) for i in value['turn_refs']]
    return out

def allowed(code,x):
    if code.startswith('PR-'):return set(x.probe)
    if code=='UP-TYPE':return set(x.probe+x.answer+x.next_move)
    return set(x.probe+x.answer)
def fill_solo_team_codes(episode,segment,codes):
    """Team codes cannot apply with one founder on the roster: record them as not applicable
    instead of asking the model (saves a prompt section and removes a compliance risk)."""
    filled=dict(episode or {})
    for code in codes:
        filled[code]={'value':None,'uncodable':True,'missing_reason':'not_applicable',
                      'evidence':'Single founder on the roster: team codes do not apply (filled by the pipeline, not rated).',
                      'turn_refs':sorted(segment.turn_refs)}
    return filled

def validate_ratings(record,segment,bundle,context=(),exchange_ids=None):
    m,t=check(bundle);xs={x.exchange_id:x for x in m.exchanges if x.segment_id==segment.segment_id and (exchange_ids is None or x.exchange_id in exchange_ids)};errors=[]
    items=record['exchanges']
    if len(items)!=len(xs) or {v['exchange_id'] for v in items}!=set(xs):return ['exchange ratings incomplete/duplicated']
    def validate(code,v,refs):
        if not v['turn_refs'] or not set(v['turn_refs'])<=refs:errors.append(code+': missing/out-of-window refs')
        if v['uncodable']!=(v['missing_reason']!='none'):errors.append(code+': missingness reason mismatch')
        if v['uncodable']!=(v['value'] is None):errors.append(code+': value/missingness mismatch')
        if v['missing_reason']=='pipeline_defect':errors.append(code+': source/map repair required')
        ts={z.turn_idx:z for z in t.turns}
        if refs and not v['uncodable']:
            lo=min(ts[i].start_s for i in refs);hi=max(ts[i].end_s for i in refs)
            if any(z.status=='unobservable' and z.start_s<hi and z.end_s>lo and (code in z.affected_codes or '*' in z.affected_codes) for z in t.issues):errors.append(code+': source observability issue requires missingness')
    for row in items:
        x=xs[row['exchange_id']]
        for code,v in row['ratings'].items():
            validate(code,v,allowed(code,x))
            if code=='UP-TYPE' and v['missing_reason']=='cutoff' and x.uptake_status!='cutoff':errors.append('UP-TYPE: cutoff contradicts frozen next-move status; source review required')
            if not v['uncodable']:
                if code=='UP-TYPE' and (x.uptake_status!='observed' or not set(v['turn_refs'])&set(x.answer) or not set(v['turn_refs'])&set(x.next_move)):errors.append('UP-TYPE: requires observed answer and next move')
                if code=='UP-TYPE' and x.next_move:
                    actors={next(t.speaker for t in t.turns if t.turn_idx==i) for i in x.next_move}
                    if x.investor not in actors and v['value']!='NONE':errors.append('UP-TYPE: different investor must be NONE')
                if code.startswith(('RS-','PREP-')) and x.answer and not set(v['turn_refs'])&set(x.answer):errors.append(code+': must cite designated answer')
                if not x.answer and code not in {'PR-FRAME','PR-SEV','RS-RESP'}:errors.append(code+': no response opportunity')
                if not x.answer and code=='RS-RESP' and v['value']!='NONE':errors.append('RS-RESP: no answer requires NONE')
    for code,v in (record['episode'] or {}).items():
        validate(code,v,set(segment.turn_refs)|(set(context) if code=='RP-RECOV' else set()))
        if sum(r['role']=='founder' for r in bundle['roster'])<2 and code.startswith('TM-') and not v['uncodable']:errors.append(code+': solo founder')
        if segment.cut and code in {'RAP-INTERRUPT','RP-RECOV'} and not v['uncodable']:errors.append(code+': cut prohibits timing inference')
    return errors

class VisualSlot(Strict):
    investor:str
    bin_index:int
    start_s:float|None
    end_s:float|None
    role:Literal['speaking','listening','unavailable']
    linkage:Literal['synchronized_speech','shared_frame','unavailable']
    evidence:str
    missing_reason:Literal['none','off_camera','ambiguous_attribution','edited_adjacency','short_interval']
class VisualInventory(Strict):
    slots:list[VisualSlot]
class VisualRating(Strict):
    investor:str
    bin_index:int
    verified:bool
    value:int|None=Field(ge=1,le=5)
    evidence:str
    missing_reason:Literal['none','off_camera','ambiguous_attribution','edited_adjacency','ambiguous_display','short_interval']
class VisualRatings(Strict):
    ratings:list[VisualRating]
def validate_visual(inv,roster,start,end):
    import math
    expected={(r['speaker'],i) for r in roster if r['role']=='investor' for i in range(math.ceil((end-start)/60))}
    if len(inv.slots)!=len(expected) or {(s.investor,s.bin_index) for s in inv.slots}!=expected:raise ValueError('visual sampling grid incomplete')
    for s in inv.slots:
        if s.start_s is None or s.end_s is None:
            if s.start_s is not None or s.end_s is not None or s.missing_reason=='none' or s.role!='unavailable' or s.linkage!='unavailable':raise ValueError('invalid missing visual interval')
        else:
            lo=start+60*s.bin_index;hi=min(end,lo+60)
            if not lo<=s.start_s<s.end_s<=hi or not 2<=s.end_s-s.start_s<=5:raise ValueError('visual interval outside bin/duration')
            if s.missing_reason!='none' or s.role=='unavailable' or s.linkage=='unavailable':raise ValueError('visual linkage unavailable')
            if (s.role=='speaking')!=(s.linkage=='synchronized_speech'):raise ValueError('role/linkage mismatch')
def validate_visual_ratings(result,inv):
    slots={(s.investor,s.bin_index):s for s in inv.slots}
    if len(result.ratings)!=len(slots) or {(r.investor,r.bin_index) for r in result.ratings}!=slots.keys():raise ValueError('visual rating grid incomplete')
    for r in result.ratings:
        s=slots[r.investor,r.bin_index]
        if (r.value is None)!=(r.missing_reason!='none'):raise ValueError('visual value/missingness mismatch')
        if r.value is not None and (s.start_s is None or not r.verified):raise ValueError('unverified interval cannot be scored')
