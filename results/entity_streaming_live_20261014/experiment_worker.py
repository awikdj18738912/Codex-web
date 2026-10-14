"""Paired old/new policy in a private frozen full-Web worker."""
import contextvars
import importlib.util
import json
import time
from pathlib import Path
import sys
import weakref
from urllib.parse import parse_qs

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'source_snapshot'))
import system.web_app as web
import system.entity_audio_repair as audio
from system.entity_store import EntityStore

meta=json.loads((OUT/'library_metadata.json').read_text())['libraries']
definitions={k:EntityStore(v['snapshot_path']).list_entities(include_disabled=True) for k,v in meta.items()}
library=contextvars.ContextVar('eval_library',default=None)
policy=contextvars.ContextVar('eval_policy',default='old')
controllers=weakref.WeakSet()
old_repair=audio.EntityAudioRepair
spec=importlib.util.spec_from_file_location('system._incremental_experiment',OUT/'candidate.py')
new=importlib.util.module_from_spec(spec);sys.modules[spec.name]=new;spec.loader.exec_module(new)

class EvaluationStore(EntityStore):
    def list_entities(self,*,domain=None,include_disabled=False):
        key=library.get()
        if key is None:return super().list_entities(domain=domain,include_disabled=include_disabled)
        return tuple(d for d in definitions[key] if (include_disabled or d.enabled) and (domain is None or d.domain in (domain,'general')))

def repair(request,definitions,**kwargs):
    cls=new.StreamingEntityRepair if policy.get()=='new' else old_repair
    measured=dict(primary_pcm_seconds=0.,replay_pcm_seconds=0.,primary_http_calls=0,
        replay_http_calls=0,primary_wall_seconds=0.,replay_wall_seconds=0.,primary_finish_text=None,
        active_http_requests=0,primary_reported_model_ms=0.,replay_reported_model_ms=0.)
    primary=None;started=0
    async def backend(endpoint,sid=None,data=b'',params=None):
        nonlocal primary,started
        is_primary=sid==primary and primary is not None
        if endpoint=='/stream/start':
            is_primary=started==0;started+=1
        prefix='primary' if is_primary else 'replay'
        measured['active_http_requests']+=1
        measured[prefix+'_http_calls']+=1
        if endpoint=='/stream/chunk':measured[prefix+'_pcm_seconds']+=len(data)/64000
        before=time.perf_counter()
        try:
            result=await request(endpoint,sid,data,params)
            measured[prefix+'_reported_model_ms']+=float((result.get('asr_timing') or {}).get('model_ms',0.) or 0.)
            if endpoint=='/stream/start' and is_primary:primary=result['session_id']
            if endpoint=='/stream/finish' and is_primary:measured['primary_finish_text']=result.get('text','')
            return result
        finally:
            measured[prefix+'_wall_seconds']+=time.perf_counter()-before
            measured['active_http_requests']-=1
    instance=cls(backend,definitions,**kwargs)
    original_snapshot=instance.snapshot
    def snapshot():
        result=original_snapshot();result['evaluation_backend']=dict(measured);return result
    instance.snapshot=snapshot
    controllers.add(instance);return instance

audio.EntityAudioRepair=repair
web.EntityStore=EvaluationStore
original=web.create_app
class Router:
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        if scope['type']=='http' and scope.get('path')=='/evaluation/quiescence':
            pending=0
            for c in tuple(controllers):
                pending+=sum(not t.done() for t in getattr(c,'_cleanup_tasks',()))
                task=getattr(c,'_review',None);pending+=bool(task and not task.done())
                shadow=getattr(c,'_streaming',None)
                if shadow:
                    pending+=sum(not t.done() for t in shadow.late_cleanup)
                    pending+=bool(shadow.task and not shadow.task.done())
            data=json.dumps({'pending_tasks':pending}).encode()
            await send({'type':'http.response.start','status':200,'headers':[(b'content-type',b'application/json')]})
            await send({'type':'http.response.body','body':data});return
        if scope['type']!='websocket':return await self.app(scope,receive,send)
        q=parse_qs(scope.get('query_string',b'').decode());key=q.get('eval_library',[None])[0]
        variant=q.get('eval_policy',[None])[0]
        if key not in definitions or variant not in {'old','new'}:
            await send({'type':'websocket.close','code':1008});return
        a=library.set(key);b=policy.set(variant)
        try:await self.app(scope,receive,send)
        finally:library.reset(a);policy.reset(b)

def create(*args,**kwargs):return Router(original(*args,**kwargs))
web.create_app=create
web.main()
