"""Versioned interaction maps and fail-closed evidence gates for Stage B.

Legacy EpisodeSpan records are readable but cannot stand in for this reviewed contract.
Turn IDs are stable, non-positional IDs; repairs retain IDs and allocate new ones.
"""
from __future__ import annotations
import hashlib
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Turn(Strict):
    turn_idx: int = Field(ge=0)
    speaker: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    tag: str

class Issue(Strict):
    start_s: float
    end_s: float
    status: Literal['resolved', 'unobservable', 'needs_repair']
    detail: str
    affected_codes: list[str]

class TimelineReview(Strict):
    turns: list[Turn]
    issues: list[Issue]
    review_notes: str

class Exchange(Strict):
    probe: list[int]
    answer: list[int]
    uptake: list[int]
    note: str

class Episode(Strict):
    episode_id: str
    topic: str
    probing_speaker: str
    turn_refs: list[int]
    opening_probe: list[int]
    initial_answer: list[int]
    first_uptake: list[int]
    minimal_backchannels: list[int] = Field(default_factory=list, description="Only investor listener tokens DURING the INITIAL founder answer that may be skipped when identifying first uptake; not later interjections.")
    uptake_status: Literal['observed', 'clip_ends', 'uncertain']
    later_exchanges: list[Exchange]
    resolution: Literal['resolved','dropped','conceded','reframed','displaced','unclear']
    cut: bool
    rationale: str

class Exclusion(Strict):
    turn_refs: list[int]
    reason: str

class InteractionMap(Strict):
    episodes: list[Episode]
    excluded: list[Exclusion]
    issues: list[Issue]
    review_notes: str

def digest(value):
    if isinstance(value, BaseModel): value=value.model_dump()
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def validate_timeline(timeline, roster, start_s, end_s):
    ids=set(); known={p['speaker'] for p in roster}
    for t in timeline.turns:
        if t.turn_idx in ids: raise ValueError('duplicate turn ID')
        ids.add(t.turn_idx)
        if t.speaker not in known: raise ValueError('unknown speaker')
        if not start_s <= t.start_s < t.end_s <= end_s: raise ValueError('turn outside source window')
    for issue in timeline.issues:
        if issue.status=='unobservable' and (not issue.affected_codes or any(c!='*' and not c.startswith(('PR-','RS-','PREP-','RP-','RAP-','TM-','UP-')) for c in issue.affected_codes)):
            raise ValueError('unobservable issue must name affected code IDs, not turn IDs')
    if any(i.status=='needs_repair' for i in timeline.issues):
        raise ValueError('timeline has unresolved repair issues')

def validate_map(mapping, timeline, roster, start_s, cutoff_s):
    validate_timeline(timeline, roster, start_s, cutoff_s)
    turns={t.turn_idx:t for t in timeline.turns}; roles={p['speaker']:p['role'] for p in roster}
    if any(i.status=='needs_repair' for i in mapping.issues): raise ValueError('map needs repair')
    assigned=set(); episode_ids=set(); previous_end=start_s
    for ep in mapping.episodes:
        if ep.episode_id in episode_ids: raise ValueError('duplicate episode ID')
        episode_ids.add(ep.episode_id)
        refs=set(ep.turn_refs)
        if not refs or not refs <= turns.keys(): raise ValueError('unknown/empty episode turns')
        if refs & assigned: raise ValueError('episodes share turns')
        assigned |= refs
        lo=min(turns[i].start_s for i in refs); hi=max(turns[i].end_s for i in refs)
        if lo < previous_end: raise ValueError('episodes overlap in time')
        previous_end=hi
        if not ep.opening_probe or not ep.initial_answer: raise ValueError('missing probe/answer')
        for field in ['opening_probe','initial_answer','first_uptake','minimal_backchannels']:
            if not set(getattr(ep,field)) <= refs: raise ValueError('component outside episode')
        if roles.get(ep.probing_speaker) != 'investor': raise ValueError('probe actor not investor')
        if any(turns[i].speaker != ep.probing_speaker for i in ep.opening_probe): raise ValueError('probe speaker mismatch')
        if any(roles[turns[i].speaker]!='founder' for i in ep.initial_answer): raise ValueError('answer not founder')
        if any(roles[turns[i].speaker]!='investor' for i in ep.first_uptake): raise ValueError('uptake not investor')
        if (ep.uptake_status=='observed') != bool(ep.first_uptake): raise ValueError('uptake observability mismatch')
        p_end=max(turns[i].start_s for i in ep.opening_probe)
        a_start=min(turns[i].start_s for i in ep.initial_answer)
        a_end=max(turns[i].start_s for i in ep.initial_answer)
        if p_end > a_start: raise ValueError('answer precedes probe')
        if ep.first_uptake and min(turns[i].start_s for i in ep.first_uptake)<a_end: raise ValueError('uptake precedes answer')
        if ep.first_uptake and len({turns[i].speaker for i in ep.first_uptake})!=1: raise ValueError('first uptake must be one investor move; mark uncertain if inseparable')
        answer_finish=max(turns[i].end_s for i in ep.initial_answer)
        if any(not a_start<=turns[i].start_s<answer_finish for i in ep.minimal_backchannels): raise ValueError('skipped backchannel outside initial answer')
        if any(roles[turns[i].speaker]!='investor' for i in ep.minimal_backchannels): raise ValueError('backchannel not investor')
        if set(ep.minimal_backchannels)&set(ep.first_uptake): raise ValueError('uptake cannot be skipped backchannel')
        intervening=[i for i in refs if roles[turns[i].speaker]=='investor' and turns[i].start_s>a_start and i not in ep.minimal_backchannels and i not in ep.opening_probe]
        if intervening:
            first=min(intervening,key=lambda i:turns[i].start_s)
            if ep.first_uptake and min(turns[i].start_s for i in ep.first_uptake)>turns[first].start_s: raise ValueError('uptake skips investor intervention')
            if a_end>turns[first].start_s: raise ValueError('initial answer crosses investor intervention')
            if not ep.first_uptake and ep.uptake_status=='clip_ends': raise ValueError('uptake exists before clip end')
        if set(ep.opening_probe)&set(ep.initial_answer) or set(ep.initial_answer)&set(ep.first_uptake): raise ValueError('component collision')
        for ex in ep.later_exchanges:
            if not set(ex.probe+ex.answer+ex.uptake) <= refs: raise ValueError('later exchange outside episode')
    for excluded in mapping.excluded:
        refs=set(excluded.turn_refs)
        if not refs or not refs <= turns.keys() or refs&assigned: raise ValueError('invalid/duplicate exclusion')
        assigned |= refs
    if assigned != turns.keys(): raise ValueError('every Q&A turn must be assigned or explicitly excluded')

def freeze_bundle(mapping, timeline, roster, source_sha256, start_s, cutoff_s, reviewer):
    validate_map(mapping,timeline,roster,start_s,cutoff_s)
    return {'schema_version':1,'timeline':timeline.model_dump(),'mapping':mapping.model_dump(),
            'roster':roster,'source_sha256':source_sha256,'start_s':start_s,'cutoff_s':cutoff_s,
            'timeline_hash':digest(timeline),'map_hash':digest(mapping),
            'reviewer':reviewer,'status':'frozen_for_development','human_validated':False}

def assert_frozen(bundle, timeline=None, mapping=None):
    if bundle.get('status')!='frozen_for_development': raise ValueError('not frozen')
    timeline=timeline or TimelineReview.model_validate(bundle['timeline'])
    mapping=mapping or InteractionMap.model_validate(bundle['mapping'])
    if digest(timeline)!=bundle['timeline_hash'] or digest(mapping)!=bundle['map_hash']:
        raise ValueError('stale freeze: timeline/map changed; regenerate dependent records')
    validate_map(mapping,timeline,bundle['roster'],bundle['start_s'],bundle['cutoff_s'])

def allowed_refs(code, ep):
    if code.startswith('PR-'): return set(ep.opening_probe)
    if code.startswith('RS-') or code in {'PREP-SUBS','PREP-COH','PREP-CTX'}:
        return set(ep.opening_probe+ep.initial_answer)
    if code=='UP-TYPE': return set(ep.initial_answer+ep.first_uptake)
    return set(ep.turn_refs)

def validate_codes(record, ep, bundle, context_refs=()):
    assert_frozen(bundle)
    if ep.model_dump() not in bundle['mapping']['episodes']:
        raise ValueError('episode differs from frozen map')
    issues=[]; solo=sum(p['role']=='founder' for p in bundle['roster'])==1
    turns={t['turn_idx']:t for t in bundle['timeline']['turns']}
    lo=min(turns[i]['start_s'] for i in ep.turn_refs); hi=max(turns[i]['end_s'] for i in ep.turn_refs)
    uncertainty=bundle['timeline']['issues']+bundle['mapping']['issues']
    for code,v in record.items():
        allowed=allowed_refs(code,ep) | (set(context_refs) if code=='RP-RECOV' else set())
        if set(v['turn_refs'])-allowed: issues.append(f'{code}: out-of-scope evidence refs')
        if solo and code.startswith('TM-') and not v['uncodable']: issues.append(f'{code}: solo founder requires uncodable')
        if code=='UP-TYPE' and ep.uptake_status!='observed' and not v['uncodable']: issues.append('UP-TYPE: unavailable uptake requires uncodable')
        if not v['uncodable'] and not v['turn_refs']: issues.append(f'{code}: codable value needs inspected refs')
        if not v['uncodable']:
            refs=set(v['turn_refs'])
            if (code.startswith('RS-') or code in {'PREP-SUBS','PREP-COH','PREP-CTX'}) and not refs&set(ep.initial_answer): issues.append(f'{code}: must cite initial answer')
            if code=='UP-TYPE' and (not refs&set(ep.first_uptake) or not refs&set(ep.initial_answer)): issues.append('UP-TYPE: must cite answer and uptake')
        for issue in uncertainty:
            if issue['status']=='unobservable' and issue['start_s']<hi and issue['end_s']>lo and (code in issue['affected_codes'] or '*' in issue['affected_codes']) and not v['uncodable']:
                issues.append(f'{code}: unresolved observability requires uncodable')
    return issues

class Claim(Strict):
    turn_ref: int
    time_s: float
    claim: str
    metric: str
    period: str
    units: str
    scope: str

class Comparison(Strict):
    first: Claim
    second: Claim
    matching_scope_established: bool
    incompatibility: str
    reconciliation: str
    counted: bool

class PitchResult(Strict):
    value: int | None = Field(ge=0)
    uncodable: bool
    coverage_complete: bool
    inspected_turn_refs: list[int]
    comparable_claims_present: bool
    comparisons: list[Comparison]
    evidence: str
    limitations: list[str]

def validate_pitch(result, eligible, unit_by_turn, founder_refs=None, turns=None):
    problems=[]
    if result.comparable_claims_present and not result.comparisons: problems.append('comparable claims require locatable pair evidence')
    if result.uncodable != (result.value is None): problems.append('value/missingness mismatch')
    if not result.coverage_complete and not result.uncodable: problems.append('incomplete coverage requires uncodable')
    if not result.uncodable and set(result.inspected_turn_refs)!=set(eligible): problems.append('incomplete eligible turn coverage')
    if not result.uncodable and result.value!=sum(p.counted for p in result.comparisons): problems.append('count/pairs mismatch')
    seen_pairs=set()
    for p in result.comparisons:
        for claim in [p.first,p.second]:
            if founder_refs is not None and claim.turn_ref not in founder_refs: problems.append('claim is not from a founder turn')
            if turns is not None and claim.turn_ref in turns:
                t=turns[claim.turn_ref]
                if not t['start_s']<=claim.time_s<=t['end_s']: problems.append('claim time outside referenced turn')
        key=tuple(sorted([p.first.turn_ref,p.second.turn_ref]))+(p.first.metric,p.first.period,p.first.scope)
        if p.counted and key in seen_pairs: problems.append('duplicate counted pair')
        if p.counted: seen_pairs.add(key)
        if p.first.turn_ref not in eligible or p.second.turn_ref not in eligible: problems.append('ineligible claim')
        if p.counted:
            if unit_by_turn.get(p.first.turn_ref)==unit_by_turn.get(p.second.turn_ref): problems.append('same-unit pair cannot count')
            if not p.matching_scope_established or not p.incompatibility.strip(): problems.append('incompatibility not established')
            for attr in ['metric','period','units','scope']:
                a=getattr(p.first,attr).strip().lower();b=getattr(p.second,attr).strip().lower()
                if a != b or a in {'','unknown','uncertain','unspecified'}: problems.append(f'counted pair has unmatched {attr}')
    return problems

class VisibleInterval(Strict):
    start_s: float
    end_s: float
    linkage_evidence: str

class VisibilityAudit(Strict):
    speaker: str
    coverage_reviewed: bool
    intervals: list[VisibleInterval]
    limitations: list[str]

def visibility_gate(audit, speaker, start_s, end_s):
    """Check codebook coverage thresholds, counting overlapping intervals only once."""
    if not audit.coverage_reviewed or audit.speaker!=speaker:return False
    spans=sorted((v.start_s,v.end_s) for v in audit.intervals)
    if any(not start_s<=a<b<=end_s for a,b in spans):return False
    merged=[]
    for a,b in spans:
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
        else:merged.append([a,b])
    duration=end_s-start_s
    return (duration>0 and sum(b-a for a,b in merged)>=duration/2
            and max((b-a for a,b in merged),default=0)>=min(5,duration))
