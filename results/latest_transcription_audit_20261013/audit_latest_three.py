"""Read-only audit of three user sessions; no new inference or production edits."""
from collections import Counter,deque
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
EXPECTED=['89674c51eddc41cf80ec31a20e2d4261','abe70c7dbfaf4612bd7db6814fca3ebf','0647559c683948849e109ea72419bda4']
sessions=[]
for line in (ROOT/'results/web/session.jsonl').open():
 r=json.loads(line)
 if r.get('trace_id') in EXPECTED:sessions.append(r)
assert [r['trace_id'] for r in sessions]==EXPECTED
traces={tid:[] for tid in EXPECTED}
p=ROOT/'results/web/session.stream_trace.jsonl'
with p.open('rb') as f:
 f.seek(max(0,p.stat().st_size-32*1024*1024))
 for l in f:
  if not any(k.encode() in l for k in EXPECTED):continue
  try:r=json.loads(l)
  except json.JSONDecodeError:continue
  if r.get('trace_id') in traces:traces[r['trace_id']].append(r)
sec=lambda r:datetime.fromisoformat(r['captured_at']).timestamp()
local=lambda r:datetime.fromisoformat(r['captured_at']).astimezone(timezone(timedelta(hours=8))).isoformat()
labels=['仅实体替换','词库辅助识别＋实体替换','原音复核＋实体替换']
summary=[]
for index,session in enumerate(sessions):
 tid=session['trace_id'];events=traces[tid];o=session['output'];chunks=[r for r in events if r['event']=='asr_chunk_timing']
 start=next(r for r in events if r['event']=='asr_session_started');end=next(r for r in reversed(events) if r['event']=='final_result')
 entity_windows=[{'source':s['source'],'model_calls':s['model_calls'],'output':s.get('output'),'outcome':s.get('outcome'),'reject_reasons':s.get('reject_reasons')} for r in events if r['event']=='window_result' for s in r.get('segments',[]) if '厉飞' in s.get('source','') and s.get('model_calls')]
 row={'trace_id':tid,'mode_label':labels[index],'finished_local':local(session),'entity_asr_context_requested':session['entity_asr_context_requested'],'entity_asr_context_applied':session['entity_asr_context_applied'],'entity_audio_repair_requested_from_recent_access_log':[False,False,True][index],
      'total_session_seconds':round(sec(end)-sec(start),3),'after_last_audio_chunk_seconds':round(sec(end)-sec(chunks[-1]),3),'audio_chunks':len(chunks),
      'asr_request_sum_seconds':round(sum((r.get('server_timing') or {}).get('request_ms',0) for r in chunks)/1000,3),
      'asr_model_sum_seconds':round(sum((r.get('server_timing') or {}).get('model_ms',0) for r in chunks)/1000,3),
      'asr_lock_wait_sum_seconds':round(sum((r.get('server_timing') or {}).get('lock_wait_ms',0) for r in chunks)/1000,6),
      'raw_target':'厉飞宇' in o['raw_text'],'raw_observed':'厉飞羽' in o['raw_text'],'final_target':'在下厉飞宇' in o['clean_text'],
      'entity_normalizations':o['entity_normalizations'],'entity_windows':entity_windows,'refiner_stats':o['refiner_session_stats'],'refiner_accepted':o['refiner_accepted'],'refiner_reject_reasons':o['refiner_reject_reasons'],
      'refinement_log_path':str(ROOT/'results/web'/session['refinement_log_file']),
      'entity_audio_status_persisted':bool(o.get('entity_audio_repair'))}
 summary.append(row)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
lines=['# 最新三次用户转录检查','', '日期：2026-10-13，Asia/Shanghai。三次均为general领域、streaming、323个一秒传输块、ASR两秒识别间隔。同一段修仙对白；没有修改代码或词库，没有重新推理。',
 '', '| 完成时间 | 配置 | ASR字形 | 最终字形 | 服务端记录总时长 | 最后一块处理后等待 |', '|---|---|---|---|---|---|']
for r in summary:
 lines.append(f"| {r['finished_local'][11:19]} | {r['mode_label']} | {'厉飞宇' if r['raw_target'] else '厉飞羽'} | 在下厉飞宇 | {r['total_session_seconds']:.2f}秒 | {r['after_last_audio_chunk_seconds']:.2f}秒 |")
lines+=['', '第一、第三次明确记录 `bounded_prose_entity_projection_v1`：模型正文为“下厉飞宇”，新规则仅改实体，原句“在下”被保留；这两个实体窗口均accepted、无拒绝原因。第二次ASR原始文本直接是“厉飞宇”，辅助提示已应用、包含5个登记词，实体改动列表为空属正常，不是实体功能未生效。',
 '', '第三次ASR原始字形仍为“厉飞羽”；这处实体的最终修改来自新正文确认，没有看到原音复核带来额外目标修正。当前持久session/trace日志未保存复核最终状态，无法确认该用户会话的具体回退原因；不能将之前实验的replay_prompt_echo当本次已确认原因。开关依据最新三条带asr_chunk_seconds=2的WebSocket访问请求，顺序为context=0/repair=0，context=1/repair=0，context=0/repair=1，与前三条会话元数据context字段一致。',
 '', '第三次总时长短于前两次，但上传等待不同。ASR chunk请求累计仅13.04、10.81、10.76秒，ASR锁等待均约0.0005秒；总时间包含浏览器、传输、请求间隔、流式精修并发等，差额不能全部归给某个模块。最后块后18.73秒对比前两次约1秒，表明第三次明显多了收尾等待，与开启原音复核相符。不会声称原音复核在本次更快。',
 '', '三次refiner_accepted均false，因为其他窗口包含semantic_content_loss/severe_content_loss并回退；不表示实体修正被撤销。第一、第三次实体窗口已接受，最终名字三次均正确。',
 '', '辅助识别还改变了其他ASR字词，例如“继续躲避”/“继续努力”，“萧县”/“消协”，以及打斗语气词。没有完整人工参考，不能仅依据名字判断整段质量更优。',
 '', '对这处实体，继续建议前置词库辅助识别或仅实体替换；本次原音复核没有额外目标收益。完整设置、时间、实际模型响应、实体修改记录见summary.json；原始分窗报告链接记录在每条refinement_log_path。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps([{'local':r['finished_local'],'mode':r['mode_label'],'total':r['total_session_seconds'],'end_wait':r['after_last_audio_chunk_seconds'],'final_target':r['final_target']} for r in summary],ensure_ascii=False,indent=2))
