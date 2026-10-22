"""Join current live processing with matched current-source counterfactuals."""
from __future__ import annotations
import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys
import unicodedata


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def lexical(text):
    return ''.join(c for c in text if unicodedata.category(c)[0] in {'L','N','M'})


def changes(before,after):
    return [dict(kind=tag,start=a,end=b,before=before[a:b],after=after[c:d])
            for tag,a,b,c,d in SequenceMatcher(None,before,after,autojunk=False).get_opcodes() if tag!='equal']


def main(args):
    out=args.output.resolve();sys.path.insert(0,str(args.source_root.resolve()))
    from system.asr_window_confidence import locate_chunks,project_window_scores
    from system.confidence_supplement import window_evidence
    from experiments.scripts.prepare_audio5_gate_review import terminal_windows
    asr=json.loads((out/'asr_final.json').read_text())
    live=json.loads((out/'current_live.final.json').read_text())
    profiles=project_window_scores(asr['confidence_projection'],final=True,
        source_chunks=[e['text'] for e in asr['all_completed_segments']])['windows']
    policies=('all_refine','native','supplement','deployed_learned')
    arms={p:json.loads((out/f'{p}.windows.json').read_text()) for p in policies}
    for policy,segments in arms.items():
        if [s['source'] for s in segments]!=[w['text'] for w in profiles]:
            raise ValueError('Frozen policy window mismatch: '+policy)
    source_equal=live['raw_text']==asr['text']
    if not source_equal:raise ValueError('Live source differs from frozen ASR; cannot claim matched windows')
    spans=live['refinement_source_spans'];decisions=live['refinement_gate_decisions']
    if [s['source_text'] for s in spans]!=[w['text'] for w in profiles] or len(decisions)!=len(profiles):
        raise ValueError('Live final partition differs from frozen current-source partition')
    trace=[json.loads(l) for l in (out/'current_live.stream_trace.jsonl').open()]
    histories={}
    for event in trace:
        if event['event']=='window_result':
            for segment in event.get('segments',[]):
                if segment.get('window_kind')!='pending_tail':
                    histories.setdefault(segment['source'],[]).append(segment)
    terminal_live,ambiguous=terminal_windows(asr['text'],live['clean_text'],profiles)
    offsets=locate_chunks(asr['text'],[e['text'] for e in asr['all_completed_segments']])
    rows=[]
    for index,profile in enumerate(profiles):
        full,native,supplement,learned=[arms[p][index] for p in policies]
        attempts=histories.get(profile['text'],[])
        if not attempts:raise ValueError('Missing actual live window trace: '+str(index))
        current=attempts[-1]
        evidence=window_evidence(profile['text'],asr['confidence_projection'])
        proposed=changes(full['baseline'],full['output'])
        current_edits=changes(current['baseline'],current['output'])
        low=min((f['score'] for f in evidence.get('fragments',[]) if f.get('score') is not None),default=None)
        audio=[event for event,(a,b) in zip(asr['all_completed_segments'],offsets)
               if a<profile['end'] and b>profile['start']]
        ctx=(current.get('window_context') or
             next((c.get('window_context') for c in reversed(current.get('model_calls',[]))
                   if c.get('window_context')),None) or
             next((c.get('window_context') for c in full.get('model_calls',[]) if c.get('window_context')),{}))
        is_change=bool(proposed);is_lexical=lexical(full['baseline'])!=lexical(full['output'])
        suggestion='uncertain'
        reason='当前处理与全量对照均为模型意见；请听音判断是否需要修改及改动是否合理。'
        if not proposed and not current_edits:
            suggestion='keep';reason='本次没有产生窗口级改动；仍需核查是否有模型漏修，不能据此自动标为正确。'
        elif is_lexical:
            suggestion='partial';reason='全量对照有词语改动，建议逐处核查口吃、填充词、改口或错字；不自动接受整窗改写。'
        elif proposed:
            reason='对照仅有标点或空白改动，可能必要也可能属于不同写法，请结合上下文判断。'
        row=dict(id=f'W{index+1:03d}',window_index=index+1,primary_policy='current_live',
            current_source_match=source_equal,raw_text=profile['text'],baseline_text=full['baseline'],
            output_all_refine=full['output'],output_native=native['output'],output_supplement=supplement['output'],
            output_learned=learned['output'],model_calls=full.get('model_calls',[]),
            native_action=native['gate']['action'],supplement_action=supplement['gate']['action'],
            learned_action=learned['gate']['action'],current_action=decisions[index]['action'],
            current_reasons=decisions[index]['reasons'],current_gate=decisions[index],
            current_window_output=current['output'],current_terminal_output=terminal_live[index],
            current_outcome=current['outcome'],current_reject_reasons=current.get('reject_reasons',[]),
            current_model_calls=current.get('model_calls',[]),current_attempt_count=len(attempts),
            current_attempt_history=attempts,current_edits=current_edits,
            full_refine_outcome=full['outcome'],full_refine_reject_reasons=full.get('reject_reasons',[]),
            score_min_fragment=low,score_status=evidence['status'],confidence_evidence=evidence,
            assistant_recommendation=suggestion,assistant_reason=reason,proposed_edits=proposed,
            lexical_change=is_lexical,miss_candidate=is_change and decisions[index]['action']!='refine',
            left_context=ctx.get('left_context',''),right_context=ctx.get('right_context',''),
            all_refine_context=next((c.get('window_context') for c in full.get('model_calls',[]) if c.get('window_context')),{}),
            audio_start=min(e['audio_start_bound'] for e in audio),audio_end=max(e['audio_end_bound'] for e in audio),
            audio_url='audio_16k.wav',raw_start=profile['start'],raw_end=profile['end'],
            terminal_alignment_ambiguous=(index>0 and ambiguous[index-1]) or (index<len(profiles)-1 and ambiguous[index]),
            human_decision=None,human_target=None,human_audio_checked=False,training_eligible=False,
            independent_acceptance=False)
        row['window_basis']=hashlib.sha256(json.dumps(dict(asr_sha256=digest(out/'asr_final.json'),
            live_final_sha256=digest(out/'current_live.final.json'),start=profile['start'],end=profile['end'],
            raw=profile['text'],baseline=full['baseline']),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        rows.append(row)
    notes=out/'assistant_review_notes.json'
    if notes.exists():
        notes=json.loads(notes.read_text())
        for row in rows:row.update(notes.get(row['id'],{}))
    with (out/'review_queue.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    (out/'review.html').write_bytes(args.template.read_bytes())
    summary=dict(windows=len(rows),current_refine=sum(r['current_action']=='refine' for r in rows),
        current_keep=sum(r['current_action'] in {'keep','skip'} for r in rows),
        current_rejected=sum(r['current_outcome']=='rejected' for r in rows),
        current_window_changed=sum(bool(r['current_edits']) for r in rows),
        all_refine_window_changed=sum(bool(r['proposed_edits']) for r in rows),
        current_skipped_counterfactual_changes=sum(r['miss_candidate'] for r in rows),
        current_skipped_counterfactual_lexical_changes=sum(r['miss_candidate'] and r['lexical_change'] for r in rows),
        matched_sources=source_equal,matched_partitions=True,primary='actual_live_current_source',
        no_semantic_content_loss_reason=not any('semantic_content_loss' in r for r in live['refiner_reject_reasons']),
        human_audio_quality_labels=0,necessary_refinement_recall=None)
    (out/'review_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# 测试音频5：当前链路逐窗核验','',
        '本次使用新ASR、当前冻结源码的实际完整文件流式链路，以及同源四策略真实模型对照。没有复用旧模型输出。','',
        f"共{len(rows)}窗；当前送审{summary['current_refine']}、跳过{summary['current_keep']}。以下均为待用户听音确认的意见。",'',
        '[浏览器核验页](http://127.0.0.1:8086/review.html)支持按编号跳转、音频回放和独立质量判断。','']
    for r in rows:
        lines += [f"## {r['id']} · 当前{r['current_action']} · {r['audio_start']:.0f}–{r['audio_end']:.0f}秒",'',
            '原ASR：'+r['raw_text'],'','规则结果：'+r['baseline_text'],'',
            '当前窗口结果：'+r['current_window_output'],'','当前终态结果（显示对齐）：'+r['current_terminal_output'],'',
            '全窗口对照：'+r['output_all_refine'],'',
            '当前原因：'+', '.join(r['current_reasons'])+'；验收：'+r['current_outcome'],'',
            '仅供核查的建议：'+r['assistant_reason'],'']
        if r.get('assistant_target'):lines += ['局部目标建议：'+r['assistant_target'],'']
        lines += ['你的必要性判断：□ 整窗需精修 □ 部分需精修 □ 应跳过 □ 待定','',
                  '当前结果评价：□ 合理 □ 部分合理 □ 有损害 □ 没改动 □ 待定','',
                  '可接受目标与备注：','']
    (out/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--template',type=Path,required=True)
    main(parser.parse_args())
