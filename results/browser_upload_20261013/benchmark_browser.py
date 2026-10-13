"""Real Chromium/file decode/WS/ASR/Refiner comparison, with optional relay delay."""
import asyncio,hashlib,json,time
from pathlib import Path
from aiohttp import web,ClientSession,WSMsgType
from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
SOURCE=ROOT/'测试音频3.mp3'
CHROME=ROOT/'.runtime/playwright_browsers/chromium-1243/chrome-linux64/chrome'
HTML={'old':(OUT/'before_index.html').read_text(),'new':(OUT/'candidate_index.html').read_text()}
CASES=[{'id':'direct_context','delay':0,'context':True,'audio':False,'order':['old','new']},
       {'id':'relay_context','delay':.04,'context':True,'audio':False,'order':['new','old']},
       {'id':'relay_audio','delay':.04,'context':False,'audio':True,'order':['old','new']}]
MONITOR=r'''
(() => {
 const Native=window.WebSocket;
 const stats=window.__browserTransferBench={frames:[],acks:[],messages:0,firstSendAt:null,finishSentAt:null,finalAt:null,final:null,ready:null};
 window.WebSocket=class extends Native {
  constructor(...args){super(...args);this.addEventListener('message',event=>{
   const data=JSON.parse(event.data);stats.messages++;
   if(data.event==='ready')stats.ready=data;
   if(data.event==='chunk_ack')stats.acks.push({index:data.chunk_index,at:performance.now()});
   if(data.event==='error')stats.error=data;
   if(data.event==='final'){stats.finalAt=performance.now();stats.final=data;}
  });this.addEventListener('close',()=>{if(stats.finalAt===null)stats.transportError='Socket closed before final';});}
  send(data){
   if(data instanceof ArrayBuffer){
    const now=performance.now();if(stats.firstSendAt===null)stats.firstSendAt=now;
    stats.frames.push({index:stats.frames.length+1,at:now,bytes:data.byteLength});
   } else {try{if(JSON.parse(data).event==='finish')stats.finishSentAt=performance.now();}catch{}}
   return super.send(data);
  }
 };
})();
'''

async def main():
 state={};records=[]
 async with ClientSession() as http:
  async def page_response(request):
   return web.Response(text=HTML[state['arm']],content_type='text/html')
  async def proxy_http(request):
   async with http.get('http://127.0.0.1:8081'+str(request.rel_url)) as response:
    return web.Response(body=await response.read(),status=response.status,headers={'Content-Type':response.headers.get('Content-Type','application/octet-stream')})
  async def proxy_ws(request):
   client=web.WebSocketResponse(max_msg_size=32*1024*1024);await client.prepare(request)
   backend=await http.ws_connect('http://127.0.0.1:8081'+str(request.rel_url),max_msg_size=32*1024*1024)
   sha=hashlib.sha256();bytes_count=0;frame_count=0;delay=state['case']['delay']
   stats={'case':state['case']['id'],'arm':state['arm'],'started':time.perf_counter()}
   records.append(stats)
   async def direction(source,destination,upload=False):
    nonlocal bytes_count,frame_count
    queue=asyncio.Queue()
    async def reader():
     nonlocal bytes_count,frame_count
     async for message in source:
      if message.type not in (WSMsgType.TEXT,WSMsgType.BINARY):break
      if upload and message.type==WSMsgType.BINARY:
       sha.update(message.data);bytes_count+=len(message.data);frame_count+=1
      await queue.put((time.perf_counter()+delay,message.type,message.data))
     await queue.put(None)
    task=asyncio.create_task(reader())
    try:
     while True:
      item=await queue.get()
      if item is None:break
      due,kind,data=item;await asyncio.sleep(max(0,due-time.perf_counter()))
      if kind==WSMsgType.BINARY:await destination.send_bytes(data)
      else:await destination.send_str(data)
    finally:
     task.cancel();await asyncio.gather(task,return_exceptions=True)
   tasks=[asyncio.create_task(direction(client,backend,True)),asyncio.create_task(direction(backend,client))]
   try:
    await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
   finally:
    for task in tasks:task.cancel()
    await asyncio.gather(*tasks,return_exceptions=True)
    await backend.close();await client.close()
    stats.update(pcm_sha256=sha.hexdigest(),audio_bytes=bytes_count,audio_frames=frame_count)
   return client
  app=web.Application();app.router.add_get('/',page_response);app.router.add_get('/ws/stream',proxy_ws);app.router.add_get('/{path:.*}',proxy_http)
  runner=web.AppRunner(app);await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start();port=site._server.sockets[0].getsockname()[1]
  plan={'browser':'Chromium headless','audio':str(SOURCE),'audio_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'cases':CASES,'HTML_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in HTML.items()},'proxy_port':port,'delay_definition':'Controlled propagation delay per direction; queued messages retain arrival-time deadlines, not a fixed per-frame bandwidth throttle. Direct case has no artificial delay.','same_actual_ASR_refiner_backend':'127.0.0.1:8081','scope':'Local real browser; does not reproduce the user desktop proxy route or prove its latency.'}
  (OUT/'browser_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
  try:
   async with async_playwright() as playwright:
    browser=await playwright.chromium.launch(headless=True,executable_path=str(CHROME),args=['--no-sandbox'])
    for case in CASES:
     for arm in case['order']:
      state.update(arm=arm,case=case);folder=OUT/'browser'/case['id']/arm;folder.mkdir(parents=True,exist_ok=True)
      page=await browser.new_page();errors=[]
      page.on('pageerror',lambda error:errors.append(str(error)))
      await page.route('https://unpkg.com/**',lambda route:route.fulfill(status=200,content_type='application/javascript',body='window.lucide={createIcons(){}};'))
      await page.add_init_script(MONITOR)
      await page.goto(f'http://127.0.0.1:{port}/');await page.wait_for_load_state('networkidle')
      await page.select_option('#mode','streaming')
      await page.locator('#entity-asr-context').set_checked(case['context'])
      await page.locator('#entity-audio-repair').set_checked(case['audio'])
      async with page.expect_file_chooser() as chooser:
       await page.click('#record')
      file=await chooser.value
      await page.evaluate('window.__browserTransferBench.chosenAt=performance.now()')
      await file.set_files(str(SOURCE))
      last=time.perf_counter();deadline=last+180
      while not await page.evaluate('window.__browserTransferBench.finalAt !== null'):
       if time.perf_counter()>deadline:raise TimeoutError('Browser case exceeded180s')
       b=await page.evaluate('({transportError:window.__browserTransferBench.transportError,error:window.__browserTransferBench.error,acks:window.__browserTransferBench.acks.length,frames:window.__browserTransferBench.frames.length})')
       if b.get('error') or b.get('transportError'):raise RuntimeError(b)
       if time.perf_counter()-last>20:print('BROWSER_PROGRESS',case['id'],arm,b,flush=True);last=time.perf_counter()
       await asyncio.sleep(.5)
      b=await page.evaluate('window.__browserTransferBench');client_timing=await page.evaluate('window.__asrLastFileTransfer || null')
      final=b['final'];last_ack=max(a['at'] for a in b['acks'])
      summary={'case':case['id'],'arm':arm,'browser_total_seconds':round((b['finalAt']-b['chosenAt'])/1000,3),'upload_seconds':round((last_ack-b['firstSendAt'])/1000,3),'tail_seconds':round((b['finalAt']-b['finishSentAt'])/1000,3),'prepare_seconds':round((b['firstSendAt']-b['chosenAt'])/1000,3),'target_corrected':'在下厉飞宇' in final['clean_text'],'trace_id':b['ready']['trace_id'],'frames':len(b['frames']),'acks':len(b['acks']),'client_timing':client_timing,'page_errors':errors,'raw_sha256':hashlib.sha256(final['raw_text'].encode()).hexdigest(),'clean_sha256':hashlib.sha256(final['clean_text'].encode()).hexdigest(),'entity_normalizations':final['entity_normalizations'],'audio_repair':final.get('entity_audio_repair')}
      assert len(b['frames'])==len(b['acks'])==323 and not errors
      (folder/'final.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n');(folder/'client_trace.json').write_text(json.dumps(b,ensure_ascii=False,indent=2)+'\n')
      (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
      await page.close();await asyncio.sleep(.2);summary['relay_capture']=records[-1]
      with (OUT/'browser_results.jsonl').open('a') as f:f.write(json.dumps(summary,ensure_ascii=False)+'\n')
      print('BROWSER_COMPLETE',json.dumps({k:summary[k] for k in ('case','arm','browser_total_seconds','upload_seconds','tail_seconds','target_corrected')},ensure_ascii=False),flush=True)
    await browser.close()
  finally:
   await runner.cleanup()

asyncio.run(main())
