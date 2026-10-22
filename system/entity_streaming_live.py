"""Bounded streaming revisions with a final check for short recordings.

Long recordings keep their bounded streaming-only finish. Short recordings
may confirm previously unreviewed final candidates in one timed audio pass.
"""
import asyncio
from collections import deque
from copy import deepcopy
from functools import lru_cache
import json
import time

from .refinement_guard import join_refined_segments
from .entity_asr_echo_guard import compact


@lru_cache(maxsize=16)
def _selector(definitions):
    from .entity_audio_repair import GenericEntitySelector
    return GenericEntitySelector(definitions)


@lru_cache(maxsize=16)
def _direct_speller(definitions):
    from .entity_direct_spelling import DirectEntitySpeller
    return DirectEntitySpeller(definitions)


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
                   request_timeout=3., enable_live=True,
                   short_finish_max_seconds=16., short_finish_wait=4.,
                   ctc_session=None, ctc_mode="off"):
        if short_finish_max_seconds < 0 or short_finish_wait < 0:
            raise ValueError('short finish limits must be nonnegative')
        if ctc_mode not in {'off', 'shadow', 'on'}:
            raise ValueError('ctc_mode must be off, shadow, or on')
        if ctc_mode != 'off' and ctc_session is None:
            raise ValueError('A CTC session is required when ctc_mode is enabled')
        self._live_ctc=ctc_session
        self._live_ctc_mode=ctc_mode
        self._live_ctc_proposal_task=None
        self._live_ctc_stats=dict(mode=ctc_mode, submitted_audio_seconds=0.,
            acoustic_events=0, proposals=0, committed=0, shadow_proposals=0,
            streaming_review_skips=0, errors=0, finish_timed_out=0,
            compute_seconds=0.)
        self._live_primary_chunk_bytes=64000
        self._live_backend=request
        self._live_lock=asyncio.Lock();self._live_primary_waiters=0
        self._live_score=candidate_score;self._live_margin=candidate_margin
        self._live_window_bytes=int(window_seconds*64000)
        self._live_cache_bytes=int(cache_seconds*64000)
        self._live_finish_wait=finish_wait;self._live_max_queue=maximum_queue
        self._live_ratio=replay_ratio;self._live_max_replay=int(maximum_replay_seconds*64000)
        self._live_request_timeout=request_timeout;self._live_enabled=enable_live
        self._live_short_finish_bytes=min(int(short_finish_max_seconds*64000),
                                         self._live_window_bytes,self._live_cache_bytes)
        self._live_short_finish_wait=short_finish_wait
        self._live_short_finishing=False
        self._live_pcm=bytearray();self._live_pcm_start=0
        self._live_reserved_bytes=0
        self._live_queue=[];self._live_worker=None;self._cleanup_tasks=set()
        self._live_deferred={};self._live_active_job=None;self._live_medium_sent=0
        self._live_active_sessions=set();self._live_results=deque(maxlen=16)
        self._live_patches=[];self._live_published=()
        self._live_examined=set();self._live_scanned=''
        self._live_previous_decoded='';self._live_stable_prefix=''
        self._live_finishing=False;self._live_closed=False
        self._live_stats=dict(rule='bounded_incremental_entity_v3_short_finish',detections=0,
            candidates=0,admitted=0,low_score_rejected=0,ambiguous_rejected=0,
            completed_windows=0,stream_revisions=0,echo_rejected=0,
            oversized_rejected=0,budget_rejected=0,queue_rejected=0,
            cache_reused=0,unmapped=0,errors=0,late_candidates_kept=0,
            detection_seconds=0.,review_wall_seconds=0.,full_audio_fallbacks=0,
            deferred_retried=0,medium_budget_rejected=0,
            short_finish_checks=0,short_finish_changes=0,short_finish_timed_out=0,
            short_finish_errors=0,short_finish_wall_seconds=0.,
            short_finish_candidates=0,short_finish_replay_seconds=0.,
            glossary_spelling_committed=0)

    def _live_snapshot(self):
        pending=bool(self._live_worker and not self._live_worker.done())
        ctc=dict(self._live_ctc_stats)
        if self._live_ctc is not None:
            ctc['acoustic_events']=self._live_ctc.event_count
            ctc['compute_seconds']=self._live_ctc.compute_seconds
            ctc['committed']=sum(p.get('evidence_source')=='ctc' for p in self._live_patches)
        return dict(self._live_stats,ctc=ctc,pending=pending,queued_windows=len(self._live_queue),
            source_scope='stable_segments_and_stable_partial_prefix',
            pending_candidates=len(self._live_deferred),
            medium_replay_seconds=self._live_medium_sent/64000,
            whole_audio_replay_enabled=bool(self._live_short_finish_bytes and self._live_short_finish_wait),
            whole_audio_replay_max_seconds=self._live_short_finish_bytes/64000,
            long_recording_whole_audio_replay_enabled=False,
            finish_starts_review=bool(self._live_short_finish_bytes and self._live_short_finish_wait),
            short_finish_max_seconds=self._live_short_finish_bytes/64000,
            short_finish_wait_seconds=self._live_short_finish_wait,
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
        # Transfer cancellation ownership to the late-response janitor.
        # A remote decode must settle before its session can be cancelled.
        if endpoint!='/stream/start':self._live_active_sessions.discard(sid)
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
                if endpoint=='/stream/start' and self._live_finishing and not self._live_short_finishing:
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
        if endpoint=='/stream/chunk' and len(data)>self._live_primary_chunk_bytes:
            # Offline/file uploads may deliver many seconds in one packet.
            # Feed the unchanged ASR in one-second transport pieces so each
            # stable event can be bound to the audio received at that point.
            result=None
            for offset in range(0,len(data),self._live_primary_chunk_bytes):
                result=await self._request_live(endpoint,session_id,
                    data[offset:offset+self._live_primary_chunk_bytes],params)
                await asyncio.sleep(0)
            if not self.changes:
                result=dict(result,completed_segments=deepcopy(list(self._segments.values())))
            return result
        if endpoint=='/stream/chunk':
            before=self._audio_size
            if self.enabled and self._live_enabled:self._live_capture(data)
            if self.enabled and self._live_ctc is not None:
                try:
                    self._live_ctc.submit_audio(data)
                    self._live_ctc_stats['submitted_audio_seconds']=self._live_ctc.audio_seconds
                except Exception:
                    self._live_ctc_stats['errors']+=1
                    self._live_ctc_stats['fallback_reason']='ctc_audio_rejected_or_backlogged'
                    self._live_ctc.close();self._live_ctc=None
            result=await self._live_primary(endpoint,self._sid or session_id,data,params)
            self._record_segments(result,before)
            if self.enabled and self._live_enabled:
                await self._live_apply_glossary(str(result.get('text','')))
                await self._live_apply_ctc(str(result.get('text','')))
                await self._live_detect(result)
                previous_revision=self.revision
                result=self._live_overlay(result)
                if self.revision!=previous_revision:await self._status()
            return self._annotate(result)
        if endpoint!='/stream/finish':
            return await self._live_primary(endpoint,self._sid or session_id,data,params)
        self._live_finishing=True
        ctc_finish=None
        if self.enabled and self._live_ctc is not None:
            ctc_finish=asyncio.create_task(self._live_ctc.finish())
        self._live_stats['late_candidates_kept']+=sum(len(j['candidates']) for j in self._live_queue)+len(self._live_deferred)
        # Unstarted jobs have not consumed an acoustic verification attempt.
        # Allow a short final pass to consider their final spelling/context.
        for job in self._live_queue:
            for c in job['candidates']:
                self._live_examined.discard((c['start'],c['end'],c['observed'],c['canonical']))
        self._live_reserved_bytes-=sum(len(j['pcm']) for j in self._live_queue)
        self._live_queue.clear()
        try:
            original=await self._live_primary(endpoint,self._sid or session_id,data,params)
            self._record_segments(original)
            if self.enabled and self._live_enabled:
                await self._live_apply_glossary(str(original.get('text','')), final=True)
            # Preserve the existing deadline for an in-flight streaming review.
            if self._live_worker and not self._live_worker.done():
                await asyncio.wait({self._live_worker},timeout=self._live_finish_wait)
            if self._live_worker and not self._live_worker.done():
                self._live_worker.cancel()
                await asyncio.gather(self._live_worker,return_exceptions=True)
            self._live_queue.clear()
            if ctc_finish is not None:
                try:
                    await ctc_finish
                    await self._live_apply_ctc(str(original.get('text','')), final=True)
                except TimeoutError:
                    self._live_ctc_stats['finish_timed_out']+=1
                except Exception:
                    self._live_ctc_stats['errors']+=1
            if (self.enabled and self._live_enabled and self._live_short_finish_wait
                    and 0 < self._audio_size <= self._live_short_finish_bytes):
                began=time.monotonic()
                try:
                    await asyncio.wait_for(self._live_review_short_final(original),
                                           timeout=self._live_short_finish_wait)
                except TimeoutError:
                    self._live_stats['short_finish_timed_out']+=1
                except Exception:
                    self._live_stats['short_finish_errors']+=1
                finally:
                    self._live_short_finishing=False
                    self._live_stats['short_finish_wall_seconds']+=time.monotonic()-began
            result=self._live_overlay(original)
            if self.enabled:
                self.phase='complete';self.reason=None if self.changes else 'no_stream_confirmed_changes'
            await self._status()
            return self._annotate(result)
        finally:self.close()

    async def _live_apply_glossary(self, raw_text, *, final=False):
        """Publish glossary-owned spellings without requiring a replay."""
        from .entity_audio_repair import normalized
        proposals = await asyncio.to_thread(
            _direct_speller(self._live_spelling_definitions).select, raw_text, final)
        # Re-evaluate boundaries on every changed hypothesis. In particular,
        # a name-like partial may become part of a longer word later.
        self._live_patches = [p for p in self._live_patches
                              if p.get('evidence_source') != 'glossary_spelling']
        for proposal in proposals:
            start, end = proposal['start'], proposal['end']
            overlaps = [p for p in self._live_patches if start < p['end'] and p['start'] < end]
            if any(p.get('evidence_source') != 'ctc' for p in overlaps):
                continue
            for patch in overlaps:
                self._live_patches.remove(patch)
            self._live_patches.append(dict(proposal,
                source_left=normalized(raw_text[max(0,start-12):start]),
                source_right=normalized(raw_text[end:end+12])))
        self._live_stats['glossary_spelling_committed'] = sum(
            p.get('evidence_source') == 'glossary_spelling' for p in self._live_patches)

    async def _live_apply_ctc(self,raw_text,*,final=False):
        """Inspect completed acoustic work without delaying ordinary ASR chunks."""
        from .entity_audio_repair import normalized
        if self._live_ctc is None or not raw_text:
            return
        if final:
            if self._live_ctc_proposal_task is not None:
                if self._live_ctc_proposal_task.done():
                    try:self._live_ctc_proposal_task.result()
                    except Exception:self._live_ctc_stats['errors']+=1
                else:self._live_ctc_proposal_task.cancel()
            self._live_ctc_proposal_task=None
            try:
                proposals=await asyncio.wait_for(
                    asyncio.to_thread(self._live_ctc.propose,raw_text,final=True),.8)
            except TimeoutError:
                self._live_ctc_stats['finish_timed_out']+=1;return
            except Exception:
                self._live_ctc_stats['errors']+=1;return
        else:
            proposals=[]
            task=self._live_ctc_proposal_task
            if task is not None and task.done():
                try:proposals=task.result()
                except Exception:self._live_ctc_stats['errors']+=1
                self._live_ctc_proposal_task=None
            if self._live_ctc_proposal_task is None:
                self._live_ctc_proposal_task=asyncio.create_task(
                    asyncio.to_thread(self._live_ctc.propose,raw_text,final=False))
        self._live_ctc_stats['proposals']+=len(proposals)
        if self._live_ctc_mode=='shadow':
            self._live_ctc_stats['shadow_proposals']+=len(proposals)
            return
        for proposal in proposals:
            start,end=proposal['start'],proposal['end']
            if not 0<=start<end<=len(raw_text) or raw_text[start:end]!=proposal['observed']:
                continue
            if any(start<p['end'] and p['start']<end for p in self._live_patches):
                continue
            patch=dict(proposal,evidence_source='ctc',
                source_left=normalized(raw_text[max(0,start-12):start]),
                source_right=normalized(raw_text[end:end+12]))
            self._live_patches.append(patch)

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
        if stable==self._live_scanned and not self._live_deferred:return
        start=max(0,_common_prefix(stable,self._live_scanned)-34)
        began=time.monotonic()
        candidates=[]
        if stable!=self._live_scanned:
            candidates,_=await asyncio.to_thread(lambda:_selector(self.definitions).select(stable[start:]))
        self._live_stats['detection_seconds']+=time.monotonic()-began
        self._live_stats['detections']+=1;self._live_scanned=stable
        candidates=[dict(c,start=c['start']+start,end=c['end']+start) for c in candidates]
        self.candidate_count=len(candidates)
        self.max_candidate_score=max((c['retrieval_score'] for c in candidates),default=0.)
        self._live_stats['candidates']+=len(candidates)
        current={(c['start'],c['end'],c['observed'],c['canonical']):c for c in candidates}
        for key,c in self._live_deferred.items():
            if stable[c['start']:c['end']]==c['observed'] and key not in current:
                current[key]=c;self._live_stats['deferred_retried']+=1
        self._live_deferred={}
        candidates=sorted(current.values(),key=lambda c:c['retrieval_score'],reverse=True)
        chosen=self._live_choose_candidates(candidates,stable,len(completed))
        if not chosen:return
        for c in chosen:
            window=self._live_window(text,stable,c)
            if window is None:continue
            key=(c['start'],c['end'],c['observed'],c['canonical'])
            left,right=window
            if self._live_cached(text,c,left,right):
                self._live_examined.add(key);continue
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
                proposal=dict(start=a,end=b,pcm=pcm,text=stable,candidates=local)
                medium_delta=(len(pcm) if self._live_medium(proposal) else 0)-(len(job['pcm']) if self._live_medium(job) else 0)
                if self._live_medium_load()+medium_delta>budget*.5:continue
                self._live_queue[index]=proposal
                self._live_reserved_bytes+=extra
                merged=True;break
            if merged:self._live_examined.add(key);continue
            if len(self._live_queue)>=self._live_max_queue:
                self._live_stats['queue_rejected']+=1;self._live_defer(c);continue
            budget=min(self._live_max_replay,max(self._live_window_bytes,int(self._audio_size*self._live_ratio)))
            if self.replay_audio_bytes+self._live_reserved_bytes+right-left>budget:
                self._live_stats['budget_rejected']+=1;self._live_defer(c);continue
            pcm=self._live_read(left,right)
            if pcm is None:self._live_stats['unmapped']+=1;continue
            job=dict(start=left,end=right,pcm=pcm,text=stable,candidates=[c])
            if self._live_medium(job) and self._live_medium_load()+len(pcm)>budget*.5:
                self._live_stats['medium_budget_rejected']+=1;self._live_defer(c);continue
            self._live_queue.append(job);self._live_examined.add(key)
            self._live_queue.sort(key=lambda j:max(c['retrieval_score'] for c in j['candidates']),reverse=True)
            self._live_reserved_bytes+=len(pcm)
        if self._live_queue and (self._live_worker is None or self._live_worker.done()):
            self._live_worker=asyncio.create_task(self._live_review_loop())

    def _live_choose_candidates(self,candidates,stable,completed_chars,*,allow_ctc_patches=False):
        """Share lexical and ambiguity guards between streaming and final text."""
        from .entity_audio_repair import normalized,_short_expansion_supported
        from .entity_direct_spelling import _function_words
        chosen=[]
        for c in candidates:
            if c['observed'] in _function_words():
                continue
            if c['retrieval_score']<self._live_score or not _short_expansion_supported(c):
                self._live_stats['low_score_rejected']+=1;continue
            rivals=[r['retrieval_score'] for r in candidates if r['canonical']!=c['canonical']
                    and (r['start'],r['end'])==(c['start'],c['end'])]
            if rivals and c['retrieval_score']-max(rivals)<self._live_margin:
                self._live_stats['ambiguous_rejected']+=1;continue
            if c['end']>completed_chars and len(normalized(stable[c['end']:]))<8:continue
            c=dict(c,source_left=normalized(stable[max(0,c['start']-12):c['start']]),
                   source_right=normalized(stable[c['end']:c['end']+12]))
            key=(c['start'],c['end'],c['observed'],c['canonical'])
            if key in self._live_examined:continue
            if any(c['canonical']==p['canonical'] and c['start']<p['end'] and p['start']<c['end']
                   and (not allow_ctc_patches or p.get('evidence_source')!='ctc')
                   for p in self._live_patches):
                if any(p.get('evidence_source')=='ctc' and c['canonical']==p['canonical']
                       and c['start']<p['end'] and p['start']<c['end'] for p in self._live_patches):
                    self._live_ctc_stats['streaming_review_skips']+=1
                continue
            if any(c['canonical']==p['canonical'] and c['start']<p['end'] and p['start']<c['end']
                   for p in chosen):continue
            chosen.append(c)
        return chosen

    async def _live_review_short_final(self,payload):
        """One additional, timed pass for final-only short-recording candidates.

        This has its own one-recording audio allowance, capped by the existing
        window/cache limits. It does not enlarge the streaming or long-audio
        replay budgets and never retries candidates already acoustically tried.
        """
        text=str(payload.get('text',''))
        if not text.strip() or len(text)>self.max_source_chars:return
        pcm=self._live_read(0,self._audio_size)
        if pcm is None:return
        candidates,_=await asyncio.to_thread(lambda:_selector(self.definitions).select(text))
        self._live_stats['short_finish_candidates']+=len(candidates)
        chosen=self._live_choose_candidates(candidates,text,len(text),allow_ctc_patches=True)
        # A fast CTC patch remains provisional for the short final review.
        # The prompted audio pass may confirm a different overlapping name.
        chosen=[c for c in chosen if not any(c['start']<p['end'] and p['start']<c['end']
                    and p.get('evidence_source')!='ctc' for p in self._live_patches)]
        if not chosen or self._live_closed:return
        terms=tuple(dict.fromkeys(c['canonical'] for c in chosen))[:8]
        chosen=[c for c in chosen if c['canonical'] in terms]
        job=dict(start=0,end=self._audio_size,pcm=pcm,text=text,candidates=chosen,short_finish=True)
        self._live_short_finishing=True
        self._live_reserved_bytes+=len(pcm)
        self._live_examined.update((c['start'],c['end'],c['observed'],c['canonical']) for c in chosen)
        changes=await self._live_review(job)
        for c in changes:
            if self._live_accept_review_change(c):
                self._live_stats['short_finish_changes']+=1

    @staticmethod
    def _live_medium(job):
        return max(c['retrieval_score'] for c in job['candidates'])<.85

    def _live_medium_load(self):
        remaining=sum(len(j['pcm']) for j in self._live_queue if self._live_medium(j))
        active=self._live_active_job
        if active and self._live_medium(active):remaining+=len(active['pcm'])-active.get('_sent',0)
        return self._live_medium_sent+remaining

    def _live_defer(self,c):
        key=(c['start'],c['end'],c['observed'],c['canonical'])
        self._live_deferred[key]=c
        ordered=sorted(self._live_deferred.items(),key=lambda item:item[1]['retrieval_score'],reverse=True)
        self._live_deferred=dict(ordered[:16])

    def _live_window(self,text,stable,c):
        from .entity_audio_repair import normalized
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
        from .entity_audio_repair import verify_entity_audio_changes
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
            job=self._live_queue.pop(0);self._live_active_job=job
            try:
                changes=await self._live_review(job)
                for c in changes:
                    self._live_accept_review_change(c)
            except asyncio.CancelledError:raise
            except Exception:self._live_stats['errors']+=1
            finally:self._live_active_job=None
            if self._live_finishing:break

    async def _live_review(self,job):
        from .entity_audio_repair import REPLAY_PREFIX,REPLAY_DECODE_SECONDS,verify_entity_audio_changes
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
            self.replay_mode='short_finish' if job.get('short_finish') else 'streaming_local'
            if job.get('short_finish'):self._live_stats['short_finish_checks']+=1
            self._live_stats['admitted']+=len(job['candidates'])
            self.replay_windows.append(dict(start_seconds=job['start']/64000,end_seconds=job['end']/64000))
            for offset in range(0,len(job['pcm']),self.replay_chunk_bytes):
                part=job['pcm'][offset:offset+self.replay_chunk_bytes]
                self.replay_audio_bytes+=len(part);sent+=len(part);job['_sent']=sent
                if job.get('short_finish'):self._live_stats['short_finish_replay_seconds']+=len(part)/64000
                if not job.get('short_finish') and self._live_medium(job):self._live_medium_sent+=len(part)
                self._live_reserved_bytes-=len(part)
                payload=await self._live_background('/stream/chunk',sid,part)
                if any(s and s in compact(str(payload.get('text',''))) for s in signatures):
                    self._live_stats['echo_rejected']+=1;return []
                for e in payload.get('completed_segments',[]):
                    if isinstance(e,dict) and isinstance(e.get('segment_id'),int):stable_events[e['segment_id']]=e
                stable=join_refined_segments(str(e.get('text','')) for _,e in sorted(stable_events.items()))
                # Short final review must include the complete recording: an
                # early confirmed name cannot prevent checking a later name.
                if stable and not job.get('short_finish'):
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

    def _live_accept_review_change(self,candidate):
        """A successful audio replay may supersede a provisional CTC patch."""
        overlaps=[p for p in self._live_patches if
                  candidate['start']<p['end'] and p['start']<candidate['end']]
        if any(p.get('evidence_source')!='ctc' for p in overlaps):
            return False
        for patch in overlaps:self._live_patches.remove(patch)
        self._live_patches.append(dict(candidate,evidence_source='audio_replay'))
        return True

    def _live_overlay(self,payload):
        from .entity_audio_repair import normalized
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
        self.changes=[dict(c,entity_type=kinds.get(c['entity_id'],''),
            match_type=('GLOSSARY_SPELLING' if c.get('evidence_source')=='glossary_spelling'
                        else 'CTC_ACOUSTIC_SUPPORTED' if c.get('evidence_source')=='ctc'
                        else 'AUDIO_REPLAY_CONFIRMED')) for c in changes]
        result.update(text=corrected,completed_segments=events,entity_audio_repair_revision=self.revision)
        if changes:
            result.update(confidence=None,confidence_calibrated=False,confidence_covers_full_state=False,
                confidence_covers_full_text=False,confidence_scope='unscored_entity_revision',
                confidence_source='entity_audio_repair',confidence_mean_logprob=None,
                confidence_min_probability=None,confidence_token_count=0)
        return result

    def _close_live(self):
        self._live_closed=True
        if self._live_ctc_proposal_task and not self._live_ctc_proposal_task.done():
            self._live_ctc_proposal_task.cancel()
        if self._live_ctc is not None:self._live_ctc.close()
        if self._live_worker and not self._live_worker.done():self._live_worker.cancel()
        self._live_reserved_bytes-=sum(len(j['pcm']) for j in self._live_queue)
        self._live_queue.clear();self._live_deferred.clear();self._live_pcm.clear()
