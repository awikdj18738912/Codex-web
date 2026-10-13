"""Report observed target behavior; no full-transcript accuracy inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

OUT=Path(__file__).parent
rows=[json.loads(l) for l in (OUT/'results.jsonl').open()]
plan=json.loads((OUT/'plan.json').read_text())
assert len(rows)==12
assert len({(r['round'],r['arm']) for r in rows})==12
summary={}
for arm,settings in plan['arms'].items():
 data=[r for r in rows if r['arm']==arm];assert len(data)==3
 summary[arm]={'label':settings['label'],'runs':len(data),
  'asr_stage_target_runs':sum(r['raw_target_count']==1 and r['raw_wrong_count']==0 for r in data),
  'final_target_runs':sum(r['target_corrected'] for r in data),'keeps_zai_xia_runs':sum(r['keeps_zai_xia'] for r in data),
  'elapsed_median_seconds':round(statistics.median(r['elapsed_seconds'] for r in data),3),
  'elapsed_range_seconds':[min(r['elapsed_seconds'] for r in data),max(r['elapsed_seconds'] for r in data)],
  'refiner_calls':[r['refiner_stats']['call_count'] for r in data],
  'entity_projection_target_runs':sum(any(c.get('canonical')=='厉飞宇' and c.get('confirmation_method')=='bounded_prose_entity_projection_v1' for c in r['normalizations']) for r in data),
  'original_audio_changes_runs':sum(bool((r['audio_repair'] or {}).get('applied')) for r in data),
  'audio_reasons':dict(Counter(r['audio_repair']['reason'] for r in data if r['audio_repair'])),
  'audio_replay_modes':dict(Counter(r['audio_repair']['replay_mode'] for r in data if r['audio_repair'])),
  'unique_raw_full_texts':len({r['raw_sha256'] for r in data}),'unique_final_full_texts':len({r['final_sha256'] for r in data})}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
lines=['# 含“厉飞宇”的真实音频：实体模式对照','',
 '使用本地 `测试音频3.mp3`（约322.7秒），每种配置3次、交错顺序，共12次实际ASR及精修会话。同一PCM哈希，一秒传输块、ASR两秒识别间隔、streaming模式；语言Chinese、领域general、精修模型、三态与学习门控、数字处理及收尾设置保持一致。',
 '', '正式启用general词库6项，初始辅助识别的5个提示包括厉飞宇；未新增实体、错误别名或语境。无词库对照临时服务不加载实体数据库，继承正式服务模型、CUDA环境及其余算法参数。没有修改生产功能代码或正式词库。',
 '', '| 配置 | ASR阶段标准字形 | 最终标准字形 | 中位总耗时 | 修正来源 |', '|---|---|---|---|---|']
for arm in plan['arms']:
 s=summary[arm]
 source={'no_library':'无','post_only':'新正文实体确认','context':'前置词库提示；无此词替换记录','audio':'新正文实体确认；原音复核未采用'}[arm]
 lines.append(f"| {s['label']} | {s['asr_stage_target_runs']}/3 | {s['final_target_runs']}/3 | {s['elapsed_median_seconds']:.2f}秒 | {source} |")
lines+=['', '这里的ASR阶段是Web返回的 `raw_text`（原音模式可能已经包含复核修改），最终阶段为 `clean_text`；本次原音模式三个会话均未采用复核结果，所以它们仍是首遍字形。所有三个有库配置最终均保留“在下厉飞宇”。',
 '', '原音复核三个会话均记录 `fallback / replay_prompt_echo`：提示词回显保护拒绝复核输出，change_count=0。候选范围不足以节省原音时长（`insufficient_audio_saving`），走322.688秒整段复核。最终修正记录均为 `bounded_prose_entity_projection_v1`，不能把成功归因于原音复核。',
 '', '对这段录音，前置词库辅助识别能直接得到标准字形，推荐优先使用；仅实体替换也能三次稳定修正，耗时相近。额外开启原音复核没有新增目标收益，约增加18秒。此前根据大测试集推荐原音复核，是总体场景建议；这条录音的证据支持不同选择。',
 '', '范围限制：一个用户指定音频、一个目标出现位置，没有人工核验完整参考，不报告CER或全段正确率，也不把3/3当跨音频泛化结果。共享GPU/ASR负载影响耗时；中位值只是本次会话墙钟时间。各轮的最终全文、模型调用统计及实体复核状态均已保存，其他字词差异不自动判为正确或错误。',
 '', '记录：`plan.json`、`results.jsonl`、`summary.json`、`round_*/{mode}/final.json`、`round_*/{mode}/events.jsonl`。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
