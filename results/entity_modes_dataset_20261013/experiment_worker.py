"""Frozen Web backend; per-session library chosen only from predeclared metadata."""
import contextvars,json,sys
from pathlib import Path
from urllib.parse import parse_qs
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'source_snapshot'))
import system.web_app as web
from system.entity_store import EntityStore
meta=json.loads((OUT/'library_metadata.json').read_text())['libraries']
definitions={k:EntityStore(v['snapshot_path']).list_entities(include_disabled=True) for k,v in meta.items()}
selected=contextvars.ContextVar('evaluation_library',default=None)
class EvaluationStore(EntityStore):
 def list_entities(self,*,domain=None,include_disabled=False):
  key=selected.get()
  if key is None:return super().list_entities(domain=domain,include_disabled=include_disabled)
  return tuple(d for d in definitions[key] if (include_disabled or d.enabled) and (domain is None or d.domain in (domain,'general')))
web.EntityStore=EvaluationStore
original=web.create_app
class LibraryRouter:
 def __init__(self,app):self.app=app
 async def __call__(self,scope,receive,send):
  if scope['type']=='websocket':
   key=parse_qs(scope.get('query_string',b'').decode()).get('eval_library',[None])[0]
   if key not in definitions:
    await send({'type':'websocket.close','code':1008});return
   token=selected.set(key)
   try:await self.app(scope,receive,send)
   finally:selected.reset(token)
  else:await self.app(scope,receive,send)
def create(*args,**kwargs):return LibraryRouter(original(*args,**kwargs))
web.create_app=create
web.main()
