"""Read-only comparison of the latest two completed main-Web recordings."""
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
import difflib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/latest_transcription_comparison_20261012'
OUT.mkdir(exist_ok=True)
recent = deque(maxlen=2)
with (ROOT / 'results/web/session.jsonl').open() as source:
    for line in source:
        try:
            recent.append(json.loads(line))
        except json.JSONDecodeError:
            pass
assert len(recent) == 2
ids = [r['trace_id'] for r in recent]
assert ids == ['a40cb174d5dd4a3b993331f320e8b687', '6bfe4bf8a5a04cba92946eadc0433b21']
trace = ROOT / 'results/web/session.stream_trace.jsonl'
with trace.open('rb') as source:
    source.seek(max(0, trace.stat().st_size - 64 * 1024 * 1024))
    data = source.read()
rows = {tid: [] for tid in ids}
for line in data.splitlines():
    if not any(tid.encode() in line for tid in ids):
        continue
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        continue
    if value.get('trace_id') in rows:
        rows[value['trace_id']].append(value)

def seconds(row):
    return datetime.fromisoformat(row['captured_at']).timestamp()

def local(row):
    return datetime.fromisoformat(row['captured_at']).astimezone(timezone(timedelta(hours=8))).strftime('%H:%M:%S')

access = (ROOT / '.runtime/services/learned-gate-on-20261012/8766.log').read_text(errors='replace').splitlines()
result = []
for recording in recent:
    tid = recording['trace_id']
    events = rows[tid]
    chunks = [r for r in events if r['event'] == 'asr_chunk_timing']
    start = next(r for r in events if r['event'] == 'asr_session_started')
    end = next(r for r in events if r['event'] == 'final_result')
    final_start = next(r for r in events if r['event'] == 'final_started')
    requests = Counter()
    for line in access:
        time_match = re.search(r'12/Oct/2026 (\d\d:\d\d:\d\d)', line)
        session_match = re.search(r'session_id=([a-zA-Z0-9]+)', line)
        if time_match and session_match and local(start) <= time_match.group(1) <= local(end):
            requests[session_match.group(1)] += 1
    quartiles = []
    for quarter in range(4):
        subset = chunks[len(chunks) * quarter // 4:len(chunks) * (quarter + 1) // 4]
        timing = [r['server_timing'] for r in subset]
        active = [r['model_ms'] for r in timing if r.get('model_ms', 0) > 1]
        quartiles.append(dict(quarter=quarter + 1, chunks=len(subset),
            mean_request_ms=statistics.mean(r['request_ms'] for r in timing),
            mean_lock_wait_ms=statistics.mean(r['lock_wait_ms'] for r in timing),
            mean_model_ms=statistics.mean(r['model_ms'] for r in timing),
            mean_active_model_ms=statistics.mean(active) if active else None))
    sources = Counter()
    for event in events:
        if event.get('event') == 'window_result':
            for segment in event.get('segments', []):
                if segment.get('model_calls'):
                    sources[segment.get('source', event.get('input'))] += 1
    output = recording['output']
    result.append(dict(trace_id=tid, started_local=local(start), completed_local=local(end),
        native_gate_mode=output.get('refinement_gate_mode'), learned_gate_scored=False,
        raw_chars=len(output['raw_text']), clean_chars=len(output['clean_text']),
        asr_segments=recording.get('asr_activity', {}).get('asr_segment_count'), audio_chunks=len(chunks),
        total_backend_seconds=seconds(end) - seconds(start),
        until_last_audio_chunk_seconds=seconds(chunks[-1]) - seconds(start),
        final_reconciliation_seconds=seconds(end) - seconds(final_start),
        refiner=output.get('refiner_session_stats'),
        asr_summed_seconds={key: sum(r['server_timing'].get(key, 0) for r in chunks) / 1000
                            for key in ('request_ms', 'model_ms', 'vad_ms', 'lock_wait_ms')},
        asr_quartiles=quartiles, ASR_sessions_during_run=dict(requests),
        modeled_window_events=sum(sources.values()), unique_modeled_sources=len(sources),
        extra_executions_of_identical_window_source=sum(v - 1 for v in sources.values()),
        cache={key: end.get(key) for key in ('window_cache_partition', 'window_cache_reused_group_count',
                  'window_cache_reprocessed_group_count', 'window_cache_reprocess_reasons')}))

a, b = [r['output'] for r in recent]
changes = [dict(operation=tag, before=a['clean_text'][i:j], after=b['clean_text'][u:v],
                before_context=a['clean_text'][max(0, i - 18):j + 18],
                after_context=b['clean_text'][max(0, u - 18):v + 18])
           for tag, i, j, u, v in difflib.SequenceMatcher(None, a['clean_text'], b['clean_text'], autojunk=False).get_opcodes()
           if tag != 'equal']
summary = dict(runs=result, raw_exact_equal=a['raw_text'] == b['raw_text'],
               cleaned_differences=changes, not_controlled_AB=True,
               audio_file_hash_not_recorded=True, services_modified=False,
               caution='Per-call ASR/Refiner durations overlap; final_call_count is a flag, not finish-stage timing.')
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
first, second = result
lines = ['# 最新两次转录比较（2026-10-12）', '',
    f"前一次 {first['started_local']}–{first['completed_local']}：原生门控 tri_state、学习门控关闭。后一次 {second['started_local']}–{second['completed_local']}：两种门控均关闭。", '',
    '| 指标 | 前一次 | 后一次 |', '|---|---:|---:|',
    f"| ASR 会话启动至最终结果 | {first['total_backend_seconds']:.2f} 秒 | {second['total_backend_seconds']:.2f} 秒 |",
    f"| 音频块 / 完成的语音段 | {first['audio_chunks']} / {first['asr_segments']} | {second['audio_chunks']} / {second['asr_segments']} |",
    f"| 原始文本字符数 | {first['raw_chars']} | {second['raw_chars']} |",
    f"| Refiner 实际调用 | {first['refiner']['call_count']} | {second['refiner']['call_count']} |",
    f"| Refiner 调用累计墙钟耗时 | {first['refiner']['completed_wall_latency_ms'] / 1000:.2f} 秒 | {second['refiner']['completed_wall_latency_ms'] / 1000:.2f} 秒 |",
    f"| ASR 模型调用累计耗时 | {first['asr_summed_seconds']['model_ms']:.2f} 秒 | {second['asr_summed_seconds']['model_ms']:.2f} 秒 |",
    f"| ASR 服务内部模型锁累计等待 | {first['asr_summed_seconds']['lock_wait_ms']:.3f} 秒 | {second['asr_summed_seconds']['lock_wait_ms']:.2f} 秒 |",
    f"| 真正最终收尾 | {first['final_reconciliation_seconds']:.3f} 秒 | {second['final_reconciliation_seconds']:.3f} 秒 |", '',
    '两次原始文本逐字相同（3010 字符）；精修结果各 3004 字符，只差两处句号/逗号，不包含字词差异。浏览器上传的文件名/音频哈希没有记录，无法额外核实上传字节完全相同。', '']
for diff in changes:
    lines.append(f"- 前：{diff['before_context']}；后：{diff['after_context']}。")
lines += ['', '## 为什么原始转写更慢', '',
    '第二次除了本文件的 ASR 会话，还有另外四个会话在同一时间段陆续请求 8766；这些其他请求与本文件重叠。前一次时间段只记录到本文件一个会话。ASR 进程用一把 model_lock 串行处理模型请求，最新一次累计等待 76.87 秒，这是原始转写变慢最直接的计时证据。Refiner 是另一个进程，不能直接持有这把锁，因此这项等待不能直接解释成 Refiner 锁阻塞。', '',
    '关闭原生门控也确实增加处理工作：Refiner 从 15 次增至 173 次，并且 ASR 与主 Web Refiner 同在 GPU0。新增精修可能增加 GPU 争用；ASR 模型累计耗时 41.69→57.19 秒。这与 GPU 争用相符，但日志不能单独区分 GPU 负载、音频段状态等全部影响，不能将两次总耗时差都当成门控的因果效果。', '',
    '另外，当前实现把分窗方式与原生门控模式绑定：tri_state 使用标点 K3 固定组；off 走另一条 VAD/动态分块路径，而且中间 update_segments 传入 final=True，默认 reprocess_rejected 也变为 True。关闭开关同时改变了分窗和拒绝结果重试行为。关闭原生门控后的日志有 100 个不同模型窗口源文本、133 次模型窗口执行，其中有 33 次相同源文本的额外执行；不是纯粹只去掉一个路由筛选器。', '',
    '日志中的 final_call_count=155 由调用参数 final=True 计数，不能解释成文件结束后调用 155 次。真实 final_started 到 final_result 只有 0.247 秒；额外工作发生在流式处理过程中。', '',
    '| 音频位置 | 前一次每块 ASR 请求均值 | 后一次每块 ASR 请求均值 | 后一次锁等待均值 |',
    '|---|---:|---:|---:|']
for old_q, new_q in zip(first['asr_quartiles'], second['asr_quartiles']):
    lines.append(f"| 第 {old_q['quarter']} 个四分之一 | {old_q['mean_request_ms']:.1f} ms | {new_q['mean_request_ms']:.1f} ms | {new_q['mean_lock_wait_ms']:.1f} ms |")
lines += ['', '后一次请求均值先从 191→203→219 ms 升高，最后回落至 165 ms；并非全程单调变慢。不能仅凭文本逐渐变长推断 ASR 会无限累积整段音频：服务会按 VAD/时长边界结束并重置识别状态。', '',
    '各模型累计耗时可以相互重叠，不能直接相加还原总时间。两次存在并发负载差异、分窗变化，属于运行诊断，不是严格控制的开关 A/B 实验。', '',
    '优先事项：先避免其他任务共享 ASR 请求队列，再将分窗/缓存/拒绝重试与原生门控开关解耦。当前仍按用户要求保持两种门控关闭；本次只分析日志，没有改动服务、模型或权重。', '',
    '依据：results/web/session.jsonl 最后两条记录、session.stream_trace.jsonl 对应 trace、ASR 8766 访问日志，以及 system/web_app.py / system/window_refinement.py / system/qwen_asr_stream_server.py。Luna 独立核对了文本、调用统计、final 标志和 ASR 锁语义。', '']
(OUT / 'REPORT.md').write_text('\n'.join(lines))
(OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(report=str(OUT / 'REPORT.md'), backend_seconds=[r['total_backend_seconds'] for r in result],
                      calls=[r['refiner']['call_count'] for r in result],
                      queue_wait_seconds=[r['asr_summed_seconds']['lock_wait_ms'] for r in result]), ensure_ascii=False))
