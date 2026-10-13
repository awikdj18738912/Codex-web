"""Experiment-only ASGI routing selects old/new entity module per serial WS."""
import importlib.util
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import system.entity_audio_repair as production
import system.web_app as web
spec=importlib.util.spec_from_file_location('system._entity_audio_before_stream',Path(__file__).parent/'before_entity_audio_repair.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
New=production.EntityAudioRepair;Old=old.EntityAudioRepair
original=web.create_app
class Router:
 def __init__(self,app):self.app=app
 async def __call__(self,scope,receive,send):
  if scope['type']=='websocket':
   arm=parse_qs(scope.get('query_string',b'').decode()).get('repair_variant',['new'])[0]
   production.EntityAudioRepair=Old if arm=='old' else New
  await self.app(scope,receive,send)
def create(*args,**kwargs):return Router(original(*args,**kwargs))
web.create_app=create
web.main()
