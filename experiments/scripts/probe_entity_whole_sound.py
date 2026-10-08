"""Post-acceptance development probe, never an independent acceptance set.

Allow all four glyphs to differ in a four-character term. Explicit ordinary
language counterexamples test whether medical context really identifies it.
"""
import json
import inspect
from collections import Counter
from sweep_entity_rules_v2 import relaxed_sound, definitions, OUT, read_jsonl, normalized
from rapidfuzz.distance import Levenshtein
from sweep_entity_rules import render, EntityProtector, candidate_boundary_issue, medical_person_slot, pronunciation_signature
from system.entity_pronunciation import _context_terms
import re


def run():
    # This exploratory variant removes only the old three-glyph bound; keep
    # the function body beside its result so the comparison is reproducible.
    body=inspect.getsource(relaxed_sound).replace('def relaxed_sound(', 'def whole_sound(').replace('>3:continue','>4:continue')
    env=dict(globals());exec(body,env);rule=env['whole_sound'];ds=definitions()
    rows=read_jsonl(OUT/'development_outputs.jsonl')
    counters=[
        dict(id='ordinary_roleplay',variant='constructed_text',reference='医生让孩子假装螃蟹练习动作。',raw_text='医生让孩子假装螃蟹练习动作。'),
        dict(id='ordinary_speech_example',variant='constructed_text',reference='医生用假装螃蟹这个例子说明表演技巧。',raw_text='医生用假装螃蟹这个例子说明表演技巧。'),
    ]
    stats={key:Counter() for key in ['development','negative_controls']};changes=[]
    for label,group in [('development',rows),('negative_controls',counters)]:
        for r in group:
            baseline=r.get('outputs',{}).get('current',r['raw_text']);output,edits=rule(baseline,ds)
            a=Levenshtein.distance(normalized(r['reference']),normalized(baseline));b=Levenshtein.distance(normalized(r['reference']),normalized(output))
            stats[label].update(samples=1,changed=int(output!=baseline),improved=int(b<a),regressed=int(b>a),error_delta=b-a)
            if output!=baseline:changes.append(dict(label=label,id=r['id'],variant=r['variant'],reference=r['reference'],baseline=baseline,output=output,edits=edits,error_delta=b-a))
    (OUT/'whole_sound_probe.json').write_text(json.dumps(dict(role='Additional development-only probe after heldout inspection; rejected if any counterexample regresses',counts={k:dict(v) for k,v in stats.items()},changes=changes,rule_body=body),ensure_ascii=False,indent=2))
    print(json.dumps({k:dict(v) for k,v in stats.items()},ensure_ascii=False),flush=True)


if __name__=='__main__':run()
