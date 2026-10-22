"""Bounded latest-snapshot workers; source progress never waits for text models."""
import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSnapshot:
    revision: int
    source: str
    final: bool
    progress: dict


class LatestSlot:
    def __init__(self):
        self.pending=None
        self.changed=asyncio.Condition()
        self.replaced=0
        self.final_seen=False

    async def put(self, value):
        async with self.changed:
            if self.final_seen:
                raise RuntimeError('Cannot replace a final snapshot')
            if self.pending is not None:
                self.replaced+=1
            self.pending=value
            self.final_seen=value.final
            self.changed.notify()

    async def get(self):
        async with self.changed:
            await self.changed.wait_for(lambda:self.pending is not None)
            value=self.pending
            self.pending=None
            return value


async def thread_call(function, *args, **kwargs):
    """Drain a running blocking call before releasing model/session ownership."""
    job=asyncio.create_task(asyncio.to_thread(function,*args,**kwargs))
    try:
        return await asyncio.shield(job)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(job)
        except Exception:
            pass
        raise


async def run_snapshot_pipeline(stream, translate, refine, publish):
    """Run three single-owner stages with only one pending value per boundary.

    ``translate(snapshot)`` returns a payload for the refine stage. ``refine``
    returns its event. Workers never publish results for superseded source
    revisions; even a skipped intermediate revision cannot replace final.
    """
    sources=LatestSlot();translations=LatestSlot()
    latest_revision=0
    stats={'source_updates':0,'translation_updates':0,'refinement_updates':0,
           'discarded_translation_results':0,'discarded_refinement_results':0}

    async def produce():
        nonlocal latest_revision
        last_source=None;finished=False
        async for source,final,progress in stream:
            if finished:
                raise RuntimeError('Source emitted text after final')
            if not isinstance(source,str) or not source.strip():
                if final:
                    raise ValueError('没有识别到语音，请检查文件和源语言。')
                await publish({'event':'progress',**progress});continue
            if source==last_source and not final:
                await publish({'event':'progress',**progress});continue
            latest_revision+=1
            snapshot=SourceSnapshot(latest_revision,source,final,dict(progress))
            last_source=source;finished=final;stats['source_updates']+=1
            await publish({'event':'source','revision':latest_revision,'source_text':source,**progress})
            await sources.put(snapshot)
        if not finished:
            raise ValueError('输入流结束但未发送final，无法确认最终文本。')

    async def translation_worker():
        while True:
            snapshot=await sources.get()
            payload=await translate(snapshot)
            stats['translation_updates']+=1
            if snapshot.revision<latest_revision:
                stats['discarded_translation_results']+=1
            else:
                await publish({'event':'translation','revision':snapshot.revision,
                    'source_text':snapshot.source,'translation':payload['translation'],
                    'translation_ms':payload['translation_ms'],
                    'translation_stats':payload.get('translation_stats',{}),**snapshot.progress})
                await translations.put(TranslatedSnapshot(snapshot,payload))
            if snapshot.final:
                return

    async def refinement_worker():
        while True:
            item=await translations.get()
            event=await refine(item.source,item.payload)
            stats['refinement_updates']+=1
            if item.source.revision<latest_revision:
                stats['discarded_refinement_results']+=1
            else:
                if item.final:
                    event={**event,'pipeline_stats':{**stats,
                        'merged_source_snapshots':sources.replaced,
                        'merged_translation_snapshots':translations.replaced,
                        'max_pending_per_stage':1}}
                await publish(event)
            if item.final:
                return

    tasks=[asyncio.create_task(produce()),asyncio.create_task(translation_worker()),
           asyncio.create_task(refinement_worker())]
    joined=asyncio.gather(*tasks)
    try:
        await asyncio.shield(joined)
    finally:
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        if joined.done() and not joined.cancelled():
            joined.exception()


@dataclass(frozen=True)
class TranslatedSnapshot:
    source: SourceSnapshot
    payload: dict

    @property
    def final(self):
        return self.source.final
