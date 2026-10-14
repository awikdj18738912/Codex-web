"""Bounded streaming entity detection and audio-confirmed revisions.

No whole-recording replay, no unprompted replay fallback, and no new review
starts at finish. Only the input hypothesis is used to retrieve frozen words.
"""
import asyncio
from collections import deque
from copy import deepcopy
from functools import lru_cache
import json
import time

from system.refinement_guard import join_refined_segments
from system.entity_asr_echo_guard import compact


@lru_cache(maxsize=16)
def _selector(definitions):
    from system.entity_audio_repair import GenericEntitySelector
    return GenericEntitySelector(definitions)


def _common_prefix(a,b):
    index=0
    for left,right in zip(a,b):
        if left!=right:break
        index+=1
    return index


class LiveEntityMixin:
    def _init_live(self, request, *, candidate_score=.63, candidate_margin=.04,
                   window_seconds=32., cache_seconds=96., finish_wait=.8,
                   maximum_queue=4, replay_ratio=.5, maximum_replay_seconds=300.,
                   request_timeout=3., enable_live=True):
        self._live_backend=request
        self._live_lock=asyncio.Lock();self._live_primary_waiters=0
        self._live_score=candidate_score;self._live_margin=candidate_margin
        self._live_window_bytes=int(window_seconds*64000)
        self._live_cache_bytes=int(cache_seconds*64000)
        self._live_finish_wait=finish_wait;self._live_max_queue=maximum_queue
        self._live_ratio=replay_ratio;self._live_max_replay=int(maximum_replay_seconds*64000)
        self._live_request_timeout=request_timeout;self._live_enabled=enable_live
        self._live_pcm=bytearray();self._live_pcm_start=0
        self._live_reserved_bytes=0
        self._live_queue=[];self._live_worker=None;self._cleanup_tasks=set()
        self._live_active_sessions=set();self._live_results=deque(maxlen=16)
        self._live_patches=[];self._live_published=()
        self._live_examined=set();self._live_scanned=''
        self._live_previous_decoded='';self._live_stable_prefix=''
        self._live_finishing=False;self._live_closed=False
        self._live_stats=dict(rule='bounded_incremental_entity_v1',detections=0,
            candidates=0,admitted=0,low_score_rejected=0,ambiguous_rejected=0,
            completed_windows=0,stream_revisions=0,echo_rejected=0,
            oversized_rejected=0,budget_rejected=0,queue_rejected=0,
            cache_reused=0,unmapped=0,errors=0,late_candidates_kept=0,
            detection_seconds=0.,review_wall_seconds=0.,full_audio_fallbacks=0)

    def _live_snapshot(self):
        pending=bool(self._live_worker and not self._live_worker.done())
        return dict(self._live_stats,pending=pending,queued_windows=len(self._live_queue),
            source_scope='stable_segments_and_stable_partial_prefix',
            whole_audio_replay_enabled=False,finish_starts_review=False,
            candidate_score=self._live_score,candidate_margin=self._live_margin,
            window_limit_seconds=self._live_window_bytes/64000,
            audio_cache_seconds=len(self._live_pcm)/64000,
            finish_wait_seconds=self._live_finish_wait)

    async def _live_primary(self,endpoint,sid=None,data=b'',params=None):
        self._live_primary_waiters+=1
        try:
            async with self._live_lock:
                return await self._live_backend(endpoint,sid,data,params)
        finally:self._live_primary_waiters-=1

    def _live_cleanup_late(self,task,endpoint,sid):
        async def cleanup():
            try:
                result=await task
                target=result.get('session_id') if endpoint=='/stream/start' else sid
                if target:
                    await self._live_backend('/stream/cancel',target)
                    self._live_active_sessions.discard(target)
            except Exception:pass
        cleanup_task=asyncio.create_task(cleanup());self._cleanup_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._cleanup_tasks.discard)

    async def _live_background(self,endpoint,sid=None,data=b'',params=None):
        while not self._live_closed:
            await asyncio.sleep(0)
            async with self._live_lock:
                if self._live_primary_waiters:continue
                if endpoint=='/stream/start' and self._live_finishing:
                    raise asyncio.CancelledError
                task=asyncio.create_task(self._live_backend(endpoint,sid,data,params))
                try:
                    result=await asyncio.wait_for(asyncio.shield(task),self._live_request_timeout)
                except (TimeoutError,asyncio.CancelledError):
                    self._live_cleanup_late(task,endpoint,sid);raise
                if endpoint=='/stream/start':self._live_active_sessions.add(result['session_id'])
                if endpoint in {'/stream/cancel','/stream/finish'}:self._live_active_sessions.discard(sid)
                return result
        raise asyncio.CancelledError

    def _live_capture(self,data):
        self._audio_size+=len(data);self._live_pcm.extend(data)
        excess=len(self._live_pcm)-self._live_cache_bytes
        if excess>0:
            del self._live_pcm[:excess];self._live_pcm_start+=excess

    def _live_read(self,start,end):
        if start<self._live_pcm_start or end>self._audio_size:return None
        left,right=start-self._live_pcm_start,end-self._live_pcm_start
        return bytes(self._live_pcm[left:right])

    async def _request_live(self,endpoint,session_id=None,data=b'',params=None):
        if endpoint=='/stream/start':
            self._params=dict(params or {});self._params.pop('context',None)
            result=await self._live_primary(endpoint,session_id,data,self._params)
            self._sid=result['session_id']
            if result.get('context_applied') is True:
                self.enabled=False;self.phase='skipped';self.reason='initial_context_already_applied'
            elif self.enabled:self.phase='capturing'
            return self._annotate(result)
        if endpoint=='/stream/cancel':
            try:return await self._live_primary(endpoint,self._sid or session_id,data,params)
            finally:self.close()
        if endpoint=='/stream/chunk':
            before=self._audio_size
            if self.enabled and self._live_enabled:self._live_capture(data)
            result=await self._live_primary(endpoint,self._sid or session_id,data,params)
            self._record_segments(result,before)
            if self.enabled and self._live_enabled:
                await self._live_detect(result)
                result=self._live_overlay(result)
            return self._annotate(result)
        if endpoint!='/stream/finish':
            return await self._live_primary(endpoint,self._sid or session_id,data,params)
        self._live_finishing=True
        self._live_stats['late_candidates_kept']+=sum(len(j['candidates']) for j in self._live_queue)
        self._live_reserved_bytes-=sum(len(j['pcm']) for j in self._live_queue)
        self._live_queue.clear()
        try:
            original=await self._live_primary(endpoint,self._sid or session_id,data,params)
            self._record_segments(original)
            # Only work already queued while receiving chunks can finish.
            # Newly discovered final-only words never trigger another pass.
            if self._live_worker and not self._live_worker.done():
                await asyncio.wait({self._live_worker},timeout=self._live_finish_wait)
            if self._live_worker and not self._live_worker.done():
                self._live_worker.cancel()
                await asyncio.gather(self._live_worker,return_exceptions=True)
            self._live_queue.clear()
            result=self._live_overlay(original)
            self.phase='complete';self.reason=None if self.changes else 'no_stream_confirmed_changes'
            return self._annotate(result)
        finally:self.close()

    def _live_source(self,payload):
        text=str(payload.get('text',''))
        completed=join_refined_segments(e['text'] for e in self._segments.values())
        if not self._timeline_valid or not text.startswith(completed):return None
        # Repeated transport packets with no decode are not independent
        # observations of an ASR hypothesis.
        model_ms=float((payload.get('asr_timing') or {}).get('model_ms',0.) or 0.)
        if text!=self._live_previous_decoded or model_ms>=1.:
            common=_common_prefix(self._live_previous_decoded,text)
            self._live_stable_prefix=text[:max(len(completed),common)]
            self._live_previous_decoded=text
        elif len(completed)>len(self._live_stable_prefix):self._live_stable_prefix=completed
        stable=self._live_stable_prefix
        if not text.startswith(stable):stable=completed
        return text,completed,stable

    async def _live_detect(self,payload):
        if self._live_finishing or self._live_closed:return
        source=self._live_source(payload)
        if source is None:self._live_stats['unmapped']+=1;return
        text,completed,stable=source
        if stable==self._live_scanned:return
        from system.entity_audio_repair import normalized,_short_expansion_supported
        start=max(0,_common_prefix(stable,self._live_scanned)-34)
        began=time.monotonic()
        candidates,_=await asyncio.to_thread(lambda:_selector(self.definitions).select(stable[start:]))
        self._live_stats['detection_seconds']+=time.monotonic()-began
        self._live_stats['detections']+=1;self._live_scanned=stable
        candidates=[dict(c,start=c['start']+start,end=c['end']+start) for c in candidates]
        self.candidate_count=len(candidates)
        self.max_candidate_score=max((c['retrieval_score'] for c in candidates),default=0.)
        self._live_stats['candidates']+=len(candidates)
        chosen=[]
        for c in candidates:
            if c['retrieval_score']<self._live_score or not _short_expansion_supported(c):
                self._live_stats['low_score_rejected']+=1;continue
            rivals=[r['retrieval_score'] for r in candidates if r['canonical']!=c['canonical']
                    and (r['start'],r['end'])==(c['start'],c['end'])]
            if rivals and c['retrieval_score']-max(rivals)<self._live_margin:
                self._live_stats['ambiguous_rejected']+=1;continue
            if c['end']>len(completed) and len(normalized(stable[c['end']:]))<8:continue
            c=dict(c,source_left=normalized(stable[max(0,c['start']-12):c['start']]),
                   source_right=normalized(stable[c['end']:c['end']+12]))
            key=(c['start'],c['end'],c['observed'],c['canonical'])
            if key in self._live_examined:continue
            if any(c['canonical']==p['canonical'] and c['start']<p['end'] and p['start']<c['end']
                   for p in self._live_patches):continue
            if any(c['canonical']==p['canonical'] and c['start']<p['end'] and p['start']<c['end']
                   for p in chosen):continue
            chosen.append(c)
        if not chosen:return
        for c in chosen:
            window=self._live_window(text,stable,c)
            if window is None:continue
            self._live_examined.add((c['start'],c['end'],c['observed'],c['canonical']))
            left,right=window
            if self._live_cached(text,c,left,right):continue
            # Combine overlapping unsent candidate windows into one replay.
            merged=False
            for index,job in enumerate(self._live_queue):
                if left>job['end']+64000 or right<job['start']-64000:continue
                a,b=min(left,job['start']),max(right,job['end'])
                local=[*job['candidates'],c]
                if b-a>self._live_window_bytes or len({v['canonical'] for v in local})>8:continue
                pcm=self._live_read(a,b)
                if pcm is None:continue
                extra=len(pcm)-len(job['pcm'])
                budget=min(self._live_max_replay,max(self._live_window_bytes,int(self._audio_size*self._live_ratio)))
                if self.replay_audio_bytes+self._live_reserved_bytes+extra>budget:continue
                self._live_queue[index]=dict(start=a,end=b,pcm=pcm,text=stable,candidates=local)
                self._live_reserved_bytes+=extra
                merged=True;break
            if merged:continue
            if len(self._live_queue)>=self._live_max_queue:
                self._live_stats['queue_rejected']+=1;continue
            budget=min(self._live_max_replay,max(self._live_window_bytes,int(self._audio_size*self._live_ratio)))
            if self.replay_audio_bytes+self._live_reserved_bytes+right-left>budget:
                self._live_stats['budget_rejected']+=1;continue
            pcm=self._live_read(left,right)
            if pcm is None:self._live_stats['unmapped']+=1;continue
            self._live_queue.append(dict(start=left,end=right,pcm=pcm,text=stable,candidates=[c]))
            self._live_reserved_bytes+=len(pcm)
        if self._live_queue and (self._live_worker is None or self._live_worker.done()):
            self._live_worker=asyncio.create_task(self._live_review_loop())

    def _live_window(self,text,stable,c):
        from system.entity_audio_repair import normalized
        spans=[];cursor=0
        for sid,event in self._segments.items():
            value=event['text'].strip();start=text.find(value,cursor)
            bounds=self._segment_audio.get(sid)
            if start<0 or bounds is None:self._live_stats['unmapped']+=1;return None
            spans.append((start,start+len(value),*bounds));cursor=start+len(value)
        completed=join_refined_segments(e['text'] for e in self._segments.values())
        if len(stable)>len(completed):
            spans.append((len(completed),len(stable),self._boundary_floor,self._audio_size))
        left,right=c['start'],c['end'];count=0
        while left and count<8:left-=1;count+=len(normalized(stable[left]))
        count=0
        while right<len(stable) and count<8:count+=len(normalized(stable[right]));right+=1
        relevant=[v for v in spans if v[0]<right and left<v[1]]
        if not relevant:self._live_stats['unmapped']+=1;return None
        a=max(0,min(v[2] for v in relevant)-128000)
        b=min(self._audio_size,max(v[3] for v in relevant)+128000)
        if b-a>self._live_window_bytes:
            self._live_stats['oversized_rejected']+=1;return None
        return a,b

    def _live_cached(self,text,c,start,end):
        from system.entity_audio_repair import verify_entity_audio_changes
        for result in self._live_results:
            if result['start']>start or result['end']<end:continue
            _,changes=verify_entity_audio_changes(text,result['text'],[c])
            if changes:self._live_patches.extend(changes)
            # A window is not replayed simply because another glossary word
            # later ranks for the same evidence; unsupported edits stay raw.
            self._live_stats['cache_reused']+=1
            return True
        return False

    async def _live_review_loop(self):
        while self._live_queue and not self._live_closed and not self._live_finishing:
            job=self._live_queue.pop(0)
            try:
                changes=await self._live_review(job)
                for c in changes:
                    if not any(c['start']<p['end'] and p['start']<c['end'] for p in self._live_patches):
                        self._live_patches.append(c)
            except asyncio.CancelledError:raise
            except Exception:self._live_stats['errors']+=1
            if self._live_finishing:break

    async def _live_review(self,job):
        from system.entity_audio_repair import REPLAY_PREFIX,REPLAY_DECODE_SECONDS,verify_entity_audio_changes
        terms=tuple(dict.fromkeys(c['canonical'] for c in job['candidates']))[:8]
        signatures=[compact(v) for v in ['参考术语仅用于辅助拼写','未在音频中听到的词不要补充','术语列表']]
        if len(terms)>=3:signatures.append(''.join(compact(v) for v in terms[:3]))
        sid=None;began=time.monotonic();stable_events={};sent=0
        try:
            ready=await self._live_background('/stream/start',params=dict(self._params,
                context=REPLAY_PREFIX+json.dumps(terms,ensure_ascii=False),chunk_seconds=REPLAY_DECODE_SECONDS))
            sid=ready['session_id']
            if ready.get('context_applied') is not True:return []
            self.checks+=1;self.terms=list(terms);self.replay_context_applied=True
            self.replay_mode='streaming_local';self._live_stats['admitted']+=len(job['candidates'])
            self.replay_windows.append(dict(start_seconds=job['start']/64000,end_seconds=job['end']/64000))
            for offset in range(0,len(job['pcm']),self.replay_chunk_bytes):
                part=job['pcm'][offset:offset+self.replay_chunk_bytes]
                self.replay_audio_bytes+=len(part);sent+=len(part)
                self._live_reserved_bytes-=len(part)
                payload=await self._live_background('/stream/chunk',sid,part)
                if any(s and s in compact(str(payload.get('text',''))) for s in signatures):
                    self._live_stats['echo_rejected']+=1;return []
                for e in payload.get('completed_segments',[]):
                    if isinstance(e,dict) and isinstance(e.get('segment_id'),int):stable_events[e['segment_id']]=e
                stable=join_refined_segments(str(e.get('text','')) for _,e in sorted(stable_events.items()))
                if stable:
                    _,confirmed=verify_entity_audio_changes(job['text'],stable,job['candidates'])
                    if confirmed:
                        self._live_stats['completed_windows']+=1
                        self._live_results.append(dict(start=job['start'],end=job['start']+offset+len(part),text=stable))
                        return confirmed
            final=await self._live_background('/stream/finish',sid)
            self._live_stats['completed_windows']+=1
            replay=str(final.get('text',''))
            if any(s and s in compact(replay) for s in signatures):
                self._live_stats['echo_rejected']+=1;return []
            self._live_results.append(dict(start=job['start'],end=job['end'],text=replay))
            _,changes=verify_entity_audio_changes(job['text'],replay,job['candidates'])
            return changes
        finally:
            self._live_reserved_bytes-=len(job['pcm'])-sent
            self._live_stats['review_wall_seconds']+=time.monotonic()-began
            if sid in self._live_active_sessions:
                async def cancel(target=sid):
                    try:await self._live_backend('/stream/cancel',target)
                    except Exception:pass
                    self._live_active_sessions.discard(target)
                task=asyncio.create_task(cancel());self._cleanup_tasks.add(task)
                task.add_done_callback(self._cleanup_tasks.discard)

    def _live_overlay(self,payload):
        from system.entity_audio_repair import normalized
        text=str(payload.get('text',''));changes=[];invalid=False
        for c in self._live_patches:
            if text[c['start']:c['end']]!=c['observed']:
                invalid=True;continue
            left=normalized(text[max(0,c['start']-12):c['start']])
            right=normalized(text[c['end']:c['end']+12])
            if not left.endswith(c.get('source_left','')) or not right.startswith(c.get('source_right','')):
                invalid=True;continue
            if any(c['start']<p['end'] and p['start']<c['end'] for p in changes):continue
            changes.append(c)
        self._live_patches=changes
        signature=tuple((c['start'],c['end'],c['canonical']) for c in changes)
        if not changes and not self._live_published:return payload
        events=deepcopy(list(self._segments.values()));source=join_refined_segments(e['text'] for e in events)
        if not text.startswith(source):
            self._live_patches=[];changes=[];signature=()
        result=deepcopy(payload);corrected=text
        for c in sorted(changes,key=lambda v:v['start'],reverse=True):
            corrected=corrected[:c['start']]+c['canonical']+corrected[c['end']:]
        cursor=0
        for e in events:
            value=e['text'].strip();start=text.find(value,cursor);end=start+len(value)
            if start<0:return payload
            local=[c for c in changes if c['start']<end and start<c['end']]
            for c in sorted(local,key=lambda v:v['start'],reverse=True):
                a,b=max(start,c['start'])-start,min(end,c['end'])-start
                insertion=c['canonical'] if start<=c['start']<end else ''
                value=value[:a]+insertion+value[b:]
            e['text']=value
            if local:e['confidence']=None;e['confidence_metadata']={}
            cursor=end
        # Check the exact committed prefix; a punctuation/ASCII join must not
        # make the event snapshot disagree with the authoritative raw text.
        expected=source
        committed=[c for c in changes if c['end']<=len(source)]
        for c in sorted(committed,key=lambda v:v['start'],reverse=True):
            expected=expected[:c['start']]+c['canonical']+expected[c['end']:]
        if any(c['start']<len(source)<c['end'] for c in changes) or join_refined_segments(e['text'] for e in events)!=expected:
            self._live_stats['unmapped']+=1;self._live_patches=[];changes=[];signature=()
            result=deepcopy(payload);corrected=text;events=deepcopy(list(self._segments.values()))
        if signature!=self._live_published:
            self._live_published=signature;self.revision+=1
            self._live_stats['stream_revisions']+=1
        kinds={d.entity_id:d.entity_type for d in self.definitions}
        self.changes=[dict(c,entity_type=kinds.get(c['entity_id'],''),match_type='AUDIO_REPLAY_CONFIRMED') for c in changes]
        result.update(text=corrected,completed_segments=events,entity_audio_repair_revision=self.revision)
        if changes:
            result.update(confidence=None,confidence_calibrated=False,confidence_covers_full_state=False,
                confidence_covers_full_text=False,confidence_scope='unscored_entity_revision',
                confidence_source='entity_audio_repair',confidence_mean_logprob=None,
                confidence_min_probability=None,confidence_token_count=0)
        return result

    def _close_live(self):
        self._live_closed=True
        if self._live_worker and not self._live_worker.done():self._live_worker.cancel()
        self._live_queue.clear();self._live_pcm.clear()
