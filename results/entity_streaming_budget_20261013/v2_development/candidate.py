"""Experimental incremental entity review; production defaults are untouched."""
import asyncio
from copy import deepcopy
from functools import lru_cache
import json
import time

from system.entity_asr_context import EntityASRContext
from system.entity_asr_echo_guard import EntityASREchoGuard
from system.entity_audio_repair import (
    EntityAudioRepair, GenericEntitySelector, REPLAY_PREFIX,
    _short_expansion_supported, normalized, verify_entity_audio_changes,
)
from system.refinement_guard import join_refined_segments


@lru_cache(maxsize=24)
def selector(definitions):
    return GenericEntitySelector(definitions)


def eligible_candidates(candidates, minimum_score=.63, minimum_margin=.04):
    """Independent candidate admission, with competing targets kept visible."""
    accepted=[]
    for c in candidates:
        others=[o['retrieval_score'] for o in candidates
                if o['canonical']!=c['canonical'] and (c['start'],c['end'])==(o['start'],o['end'])]
        margin=c['retrieval_score']-max(others) if others else 1.0
        if c['retrieval_score']>=minimum_score and margin>=minimum_margin and _short_expansion_supported(c):
            accepted.append(dict(c, candidate_margin=round(margin,6)))
    return accepted


class IncrementalEntityRepair(EntityAudioRepair):
    def __init__(self, request, definitions, *, candidate_score=.63, candidate_margin=.04,
                 finish_wait=.8, window_seconds=20., replay_budget_seconds=60., **kwargs):
        kwargs['streaming_replay']=False
        super().__init__(request,definitions,**kwargs)
        self._backend=request
        self._request=self._foreground_request
        self._lock=asyncio.Lock();self._foreground_waiters=0
        self._review=None;self._examined=set();self._proposals=[];self._published=()
        self._last_detection=();self._cleanup_tasks=set();self._active_replays=set()
        self._scanned_source=''
        self._reviewed_ranges=set()
        self._score=candidate_score;self._margin=candidate_margin
        self._finish_wait=finish_wait;self._window_seconds=window_seconds
        self._replay_budget=replay_budget_seconds;self._closed=False
        self._inc=dict(rule='incremental_segment_budget_v1',detections=0,admitted=0,
            rejected=0,windows=0,completed_windows=0,stream_revisions=0,
            budget_exhausted=0,unmapped=0,oversized_windows=0,errors=0,
            detection_seconds=0.,replay_wall_seconds=0.)

    def snapshot(self):
        result=super().snapshot()
        if hasattr(self,'_inc'):
            result['incremental']=dict(self._inc,pending=bool(self._review and not self._review.done()),
                candidate_score=self._score,candidate_margin=self._margin,
                finish_wait_seconds=self._finish_wait,window_limit_seconds=self._window_seconds)
        return result

    async def _foreground_request(self, endpoint, session_id=None, data=b'', params=None):
        self._foreground_waiters+=1
        try:
            async with self._lock:return await self._backend(endpoint,session_id,data,params)
        finally:self._foreground_waiters-=1

    def _cleanup(self, pending, endpoint, sid):
        async def later():
            try:
                payload=await pending
                target=payload.get('session_id') if endpoint=='/stream/start' else sid
                if target:
                    await asyncio.wait_for(self._backend('/stream/cancel',target),1.)
                    self._active_replays.discard(target)
            except Exception:pass
        task=asyncio.create_task(later());self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def _background_request(self, endpoint, session_id=None, data=b'', params=None):
        while True:
            await asyncio.sleep(0)
            async with self._lock:
                if not self._foreground_waiters:
                    task=asyncio.create_task(self._backend(endpoint,session_id,data,params))
                    try:result=await asyncio.wait_for(asyncio.shield(task),2.)
                    except (TimeoutError,asyncio.CancelledError):
                        self._cleanup(task,endpoint,session_id);raise
                    if endpoint=='/stream/start':self._active_replays.add(result['session_id'])
                    elif endpoint in {'/stream/finish','/stream/cancel'}:self._active_replays.discard(session_id)
                    return result
            await asyncio.sleep(.001)

    async def request(self, endpoint, session_id=None, data=b'', params=None):
        result=await super().request(endpoint,session_id,data,params)
        if endpoint=='/stream/chunk' and self.enabled:
            self._drain_completed()
            await self._schedule()
            result=self._overlay(result)
            return self._annotate(result)
        return result

    def _drain_completed(self):
        if not self._review or not self._review.done():return
        task,self._review=self._review,None
        try:
            for c in task.result():
                if not any(c['start']<o['end'] and o['start']<c['end'] for o in self._proposals):
                    self._proposals.append(c)
        except asyncio.CancelledError:pass
        except Exception:self._inc['errors']+=1

    async def _schedule(self):
        if (self._closed or not self.enabled or not self._timeline_valid or self._review
                or not self._segments or self._audio is None):return
        signature=tuple((k,v['text']) for k,v in self._segments.items())
        if signature==self._last_detection:return
        self._last_detection=signature
        text=join_refined_segments(e['text'] for e in self._segments.values())
        if len(text)>self.max_source_chars:return
        if not text.startswith(self._scanned_source):
            self._inc['unmapped']+=1;return
        # Scan new stable speech plus a bounded overlap, rather than repeatedly
        # searching the entire transcript as the recording grows.
        query_start=max(0,len(self._scanned_source)-34)
        before=time.monotonic()
        candidates,_=await asyncio.to_thread(lambda:selector(self.definitions).select(text[query_start:]))
        candidates=[dict(c,start=c['start']+query_start,end=c['end']+query_start) for c in candidates]
        self._scanned_source=text
        self._inc['detection_seconds']+=time.monotonic()-before;self._inc['detections']+=1
        key=lambda c:(c['start'],c['end'],c['canonical'])
        fresh=[c for c in candidates if key(c) not in self._examined]
        chosen=eligible_candidates(fresh,self._score,self._margin)
        self._inc['rejected']+=len(fresh)-len(chosen)
        self.candidate_count=len(candidates)
        self.max_candidate_score=max((c['retrieval_score'] for c in candidates),default=0.)
        if not chosen:return
        windows=self._windows(text,chosen)
        if not windows:return
        start,end,local=windows[0]
        if self.replay_audio_bytes+end-start>self._replay_budget*64000:
            self._inc['budget_exhausted']+=1;return
        # Freeze bytes before any await: capture of later chunks must not move
        # the shared spool cursor while the replay reads its own immutable PCM.
        self._audio.seek(start);pcm=self._audio.read(end-start)
        if len(pcm)!=end-start:self._inc['unmapped']+=1;return
        self._examined.update(key(c) for c in local)
        self._reviewed_ranges.add((start,end))
        self._inc['admitted']+=len(local);self._inc['windows']+=1
        self._review=asyncio.create_task(self._review_window(text,local,pcm,start,end))

    def _windows(self,text,candidates):
        events=list(self._segments.values())
        if join_refined_segments(e['text'] for e in events)!=text.strip():
            self._inc['unmapped']+=1;return []
        spans=[];cursor=0
        for e in events:
            value=e['text'].strip();start=text.find(value,cursor)
            bounds=self._segment_audio.get(e['segment_id'])
            if start<0 or not bounds:self._inc['unmapped']+=1;return []
            spans.append((start,start+len(value),*bounds));cursor=start+len(value)
        windows=[]
        for c in candidates:
            left,right=c['start'],c['end'];count=0
            while left and count<8:left-=1;count+=len(normalized(text[left]))
            count=0
            while right<len(text) and count<8:count+=len(normalized(text[right]));right+=1
            relevant=[s for s in spans if s[0]<right and left<s[1]]
            if not relevant:continue
            start=max(0,min(s[2] for s in relevant)-128000)
            end=min(self._audio_size,max(s[3] for s in relevant)+128000)
            if (start,end) in self._reviewed_ranges:continue
            if end-start>self._window_seconds*64000:
                self._inc['oversized_windows']+=1;continue
            found=next((w for w in windows if w[0]==start and w[1]==end),None)
            if found:found[2].append(c)
            else:windows.append([start,end,[c]])
        return sorted(windows,key=lambda w:(-max(c['retrieval_score'] for c in w[2]),w[0]))

    async def _review_window(self,text,candidates,pcm,start,end):
        terms=list(dict.fromkeys(c['canonical'] for c in candidates))[:8]
        context=EntityASRContext(REPLAY_PREFIX+json.dumps(terms,ensure_ascii=False),tuple(terms))
        guard=EntityASREchoGuard(self._background_request,context)
        ready=None;began=time.monotonic()
        try:
            ready=await guard.request('/stream/start',params=dict(self._params,context=context.text))
            if not ready.get('context_applied'):return []
            self.checks+=1;self.replay_context_applied=True;self.terms=terms
            self.replay_mode='incremental';self.replay_windows.append(dict(start_seconds=start/64000,end_seconds=end/64000))
            for offset in range(0,len(pcm),self.replay_chunk_bytes):
                data=pcm[offset:offset+self.replay_chunk_bytes];self.replay_audio_bytes+=len(data)
                await guard.request('/stream/chunk',ready['session_id'],data)
            replay=await guard.request('/stream/finish',ready['session_id'])
            self._inc['completed_windows']+=1
            if guard.fallback_active:return []
            _,changes=verify_entity_audio_changes(text,str(replay.get('text','')),candidates)
            return changes
        finally:
            self._inc['replay_wall_seconds']+=time.monotonic()-began
            if ready:
                sid=ready['session_id']
                if sid in self._active_replays:
                    task=asyncio.create_task(self._backend('/stream/cancel',sid))
                    self._cleanup_tasks.add(task);task.add_done_callback(self._cleanup_tasks.discard)
                    self._active_replays.discard(sid)
            guard.close()

    def _overlay(self,payload):
        text=str(payload.get('text',''));source=join_refined_segments(e['text'] for e in self._segments.values())
        proposals={(c['start'],c['end'],c['canonical']):c for c in self._proposals}
        changes=sorted(proposals.values(),key=lambda c:c['start'])
        if not changes:return payload
        if not text.startswith(source) or any(text[c['start']:c['end']]!=c['observed'] for c in changes):
            # Withdraw invalid accumulated edits with an explicit new revision.
            self._proposals=[];self._published=();self.changes=[];self.revision+=1
            result=deepcopy(payload);result['completed_segments']=deepcopy(list(self._segments.values()))
            result['entity_audio_repair_revision']=self.revision
            return result
        if any(a['end']>b['start'] for a,b in zip(changes,changes[1:])):
            self._inc['errors']+=1;return payload
        result=deepcopy(payload);corrected=text
        for c in reversed(changes):corrected=corrected[:c['start']]+c['canonical']+corrected[c['end']:]
        events=deepcopy(list(self._segments.values()));cursor=0
        for e in events:
            value=e['text'].strip();start=text.find(value,cursor);end=start+len(value)
            if start<0:return payload
            local=[c for c in changes if start<=c['start']<c['end']<=end]
            for c in reversed(local):
                x,y=c['start']-start,c['end']-start;value=value[:x]+c['canonical']+value[y:]
            e['text']=value
            if local:e['confidence']=None;e['confidence_metadata']={}
            cursor=end
        signature=tuple(proposals)
        if signature!=self._published:
            self._published=signature;self.revision+=1;self._inc['stream_revisions']+=1
        kinds={d.entity_id:d.entity_type for d in self.definitions}
        self.changes=[dict(c,entity_type=kinds.get(c['entity_id'],''),match_type='AUDIO_REPLAY_CONFIRMED') for c in changes]
        result.update(text=corrected,completed_segments=events,entity_audio_repair_revision=self.revision,
            confidence=None,confidence_calibrated=False,confidence_covers_full_state=False,
            confidence_covers_full_text=False,confidence_scope='unscored_entity_revision',
            confidence_source='entity_audio_repair',confidence_mean_logprob=None,
            confidence_min_probability=None,confidence_token_count=0)
        return result

    async def _repair(self,original):
        self._drain_completed();await self._schedule()
        deadline=time.monotonic()+self._finish_wait
        while self._review:
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            await asyncio.wait({self._review},timeout=remaining)
            if not self._review.done():break
            self._drain_completed()
            # A new detection may be needed after an earlier review completed.
            await self._schedule()
        if self._review:
            self._inc['budget_exhausted']+=1;self._review.cancel()
            await asyncio.gather(self._review,return_exceptions=True);self._review=None
        result=self._overlay(original)
        self.phase='complete';self.reason=None if self.changes else 'incremental_keep_original'
        return result

    def close(self):
        self._closed=True
        if self._review and not self._review.done():self._review.cancel()
        super().close()
