"""Private streaming wrapper; all audio confirmation work is bounded locally."""
import importlib.util
from pathlib import Path
import sys

from system.entity_audio_repair import EntityAudioRepair

spec=importlib.util.spec_from_file_location('system._live_entity_core',Path(__file__).with_name('live_core.py'))
live=importlib.util.module_from_spec(spec);sys.modules[spec.name]=live;spec.loader.exec_module(live)


class StreamingEntityRepair(live.LiveEntityMixin,EntityAudioRepair):
    def __init__(self,request,definitions,**kwargs):
        options={k:kwargs.pop(k) for k in ['candidate_score','candidate_margin','window_seconds',
            'cache_seconds','finish_wait','maximum_queue','replay_ratio','maximum_replay_seconds',
            'request_timeout','enable_live'] if k in kwargs}
        kwargs['streaming_replay']=False
        super().__init__(request,definitions,**kwargs)
        self._init_live(request,**options)

    async def request(self,endpoint,session_id=None,data=b'',params=None):
        return await self._request_live(endpoint,session_id,data,params)

    def snapshot(self):
        result=super().snapshot()
        if hasattr(self,'_live_stats'):result['incremental']=self._live_snapshot()
        return result

    def close(self):
        self._close_live()
        super().close()
