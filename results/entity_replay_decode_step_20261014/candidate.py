"""Experiment: keep first-pass ASR and entity decisions, reduce replay decoding.

The review session consumes already captured audio. Its intermediate hypothesis
is not published, so it need not decode at the first-pass display frequency.
"""
import math

from system.entity_audio_repair import EntityAudioRepair


class EfficientEntityRepair(EntityAudioRepair):
    def __init__(self, request, definitions, *, review_decode_seconds=10., **kwargs):
        if not math.isfinite(review_decode_seconds) or review_decode_seconds<=0:
            raise ValueError('review_decode_seconds must be finite and positive')
        self._review_step=review_decode_seconds
        self._primary_started=False
        self._initial_params=None
        self._review_starts=0

        async def adapter(endpoint, sid=None, data=b'', params=None):
            if endpoint=='/stream/start':
                if not self._primary_started:
                    self._primary_started=True
                    self._initial_params=dict(params or {})
                else:
                    params=dict(params or {})
                    params['chunk_seconds']=self._review_step
                    self._review_starts+=1
            return await request(endpoint, sid, data, params)

        super().__init__(adapter, definitions, **kwargs)

    def snapshot(self):
        result=super().snapshot()
        result.update(review_decode_policy='coarser_review_v7',
            review_decode_seconds=self._review_step,review_start_requests=self._review_starts,
            initial_chunk_seconds=(self._initial_params or {}).get('chunk_seconds'))
        return result
