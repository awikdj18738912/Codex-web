"""Offline, weakly supervised before-refiner gate from frozen correction pairs."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('PREGATE_DATA', '/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261026_provisional'))
ENCODER = Path(os.environ.get('PREGATE_ENCODER', str(ROOT / 'models/entity_ranker/encoder')))
OUT = Path(os.environ.get('PREGATE_OUTPUT', '/home/aim0/data/ASR/training/refiner13_pregate_tiny_20261027'))
SEED = 20261027


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def metrics(rows, probs, threshold):
    pred = np.asarray(probs) >= threshold
    labels = np.asarray([r['label'] for r in rows], dtype=bool)
    positive = int(labels.sum()); negative = len(rows) - positive
    byclass = {}
    for c in sorted({r['category'] for r in rows}):
        idx = np.array([r['category'] == c and r['label'] == 1 for r in rows])
        if idx.any():
            byclass[c] = float(pred[idx].mean())
    return {'n':len(rows), 'need_refine_count':positive, 'no_change_count':negative,
        'need_refine_recall':float(pred[labels].mean()) if positive else None,
        'missed_refine_count':int((~pred & labels).sum()),
        'clean_skip_rate':float((~pred[~labels]).mean()) if negative else None,
        'clean_sent_to_refiner':int((pred & ~labels).sum()),
        'all_skip_rate':float((~pred).mean()), 'per_class_positive_recall':byclass}


def select_threshold(rows, probs):
    # Keep recall first; if no useful operating point exists, threshold zero
    # sends everything to the Refiner and is explicitly reported as no saving.
    choices=[]
    for t in np.linspace(0,1,201):
        m=metrics(rows, probs, float(t))
        if m['need_refine_recall'] >= .99 and min(m['per_class_positive_recall'].values()) >= .98:
            choices.append((m['clean_skip_rate'], float(t), m))
    if not choices:
        return 0., metrics(rows, probs, 0.)
    _,t,m=max(choices,key=lambda item:(item[0],item[1]))
    return t,m


def prepare():
    manifest=json.loads((DATA/'manifest.json').read_text())
    for name,expected in manifest['files_sha256'].items():
        if hashlib.sha256((DATA/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen source changed: '+name)
    source=[json.loads(s) for s in (DATA/'split_audit.jsonl').open()]
    candidate=[]
    for r in source:
        candidate.append({'id':r['id'], 'split':r['split'], 'text':r['input'],
            'label':int(r['input']!=r['reference']), 'category':r['category'], 'source':r['source'],
            'derived_clean':False})
        if r['input']!=r['reference']:
            candidate.append({'id':r['id']+':clean_reference','split':r['split'],
                'text':r['reference'], 'label':0, 'category':r['category'],
                'source':r['source'], 'derived_clean':True})
    seen=defaultdict(list)
    for r in candidate: seen[r['text']].append(r)
    accepted=[]; rejected=[]
    for text,group in seen.items():
        if len({r['split'] for r in group})>1 or len({r['label'] for r in group})>1:
            rejected.extend(group)
        else:
            # Prefer original rows to derived clean references when identical.
            accepted.append(min(group,key=lambda r:r['derived_clean']))
    splits={s:[r for r in accepted if r['split']==s] for s in ('train','dev','test')}
    for s,rows in splits.items():
        (OUT/(s+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    write('manifest.json',{'source_manifest':str(DATA/'manifest.json'),
        'source_sha256':hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
        'label_rule':'input differs from candidate reference => 1; otherwise 0. Clean references also yield label 0 within the same source split.',
        'weak_labels_not_human_gold':True,'model_input':'text only; no category, source, reference or future Refiner output',
        'counts':{s:dict(Counter(str(r['label']) for r in rows)) for s,rows in splits.items()},
        'excluded_conflicting_or_cross_split_rows':len(rejected),
        'original_test_previously_inspected':True,'online_enabled':False})
    return splits,source


class Gate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(ENCODER,local_files_only=True,add_pooling_layer=False)
        self.head=torch.nn.Sequential(torch.nn.Dropout(.1),torch.nn.Linear(self.encoder.config.hidden_size,2))

    def forward(self, **inputs):
        x=self.encoder(**inputs).last_hidden_state
        mask=inputs['attention_mask'].unsqueeze(-1).to(x.dtype)
        return self.head((x*mask).sum(1)/mask.sum(1).clamp_min(1))


def main():
    torch.set_num_threads(8)
    device = torch.device(os.environ.get('PREGATE_DEVICE', 'cpu'))
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    OUT.mkdir(parents=True,exist_ok=False)
    splits,source=prepare()
    tokenizer=AutoTokenizer.from_pretrained(ENCODER,local_files_only=True)
    model=Gate().to(device)
    parameter_count=sum(p.numel() for p in model.parameters())
    print('PARAMETERS',parameter_count,'SPLITS',{s:len(r) for s,r in splits.items()},flush=True)
    encoded={}
    overflow={}
    for s,rows in splits.items():
        encoded[s]=tokenizer([r['text'] for r in rows],truncation=False)
        overflow[s]=[i for i,x in enumerate(encoded[s]['input_ids']) if len(x)>512]
    write('config.json',{'encoder':str(ENCODER),'encoder_sha256':hashlib.sha256((ENCODER/'pytorch_model.bin').read_bytes()).hexdigest(),
        'parameter_count':parameter_count,'device':str(device),'cpu_threads':8,'epochs':2,'batch_size':64,
        'lr_encoder':2e-5,'lr_head':2e-4,'positive_recall_target_dev':.99,
        'per_class_recall_target_dev':.98,'overflow_counts':{s:len(v) for s,v in overflow.items()},
        'overflow_policy':'Skip training overflow; route to Refiner unconditionally at inference',
        'online_enabled':False})
    def batch(split,indices):
        values=[{k:v[i] for k,v in encoded[split].items()} for i in indices]
        return {k:v.to(device) for k,v in tokenizer.pad(values,padding=True,return_tensors='pt').items()}
    def predict(split):
        model.eval();values=np.ones(len(splits[split]),dtype=float)
        indices=[i for i in range(len(values)) if i not in overflow[split]]
        with torch.inference_mode():
            for start in range(0,len(indices),64):
                ids=indices[start:start+64]
                values[ids]=model(**batch(split,ids)).softmax(-1)[:,1].cpu().numpy()
        return values
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':2e-5},
        {'params':model.head.parameters(),'lr':2e-4}],weight_decay=.01)
    labels=torch.tensor([r['label'] for r in splits['train']],dtype=torch.long,device=device)
    rng=random.Random(SEED);best=float('inf');history=[]
    started=time.perf_counter()
    for epoch in range(1,3):
        model.train();order=[i for i in range(len(labels)) if i not in overflow['train']]
        rng.shuffle(order);total=0.;n=0
        for step,start in enumerate(range(0,len(order),64),1):
            ids=order[start:start+64];optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(**batch('train',ids)),labels[ids])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())*len(ids);n+=len(ids)
            if step%100==0: print('TRAIN',epoch,step,'loss',round(total/n,5),flush=True)
        probs=predict('dev');y=np.array([r['label'] for r in splits['dev']])
        val=float(-(y*np.log(np.clip(probs,1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-probs,1e-7,1-1e-7))).mean())
        history.append({'epoch':epoch,'train_loss':total/n,'validation_loss':val})
        write('history.json',history)
        if val<best:
            best=val
            torch.save(model.state_dict(),OUT/'gate.pt')
            write('best_checkpoint.json',{'epoch':epoch,'validation_loss':val})
        print('EPOCH',history[-1],flush=True)
    model.load_state_dict(torch.load(OUT/'gate.pt',map_location='cpu',weights_only=True))
    dev_probs=predict('dev');threshold,dev_metrics=select_threshold(splits['dev'],dev_probs)
    test_probs=predict('test');test_metrics=metrics(splits['test'],test_probs,threshold)
    (OUT/'test_predictions.jsonl').write_text(''.join(json.dumps(dict(r,
        probability=float(p),route_to_refiner=bool(p>=threshold)),ensure_ascii=False)+'\n'
        for r,p in zip(splits['test'],test_probs)))
    positive_indices=[i for i,r in enumerate(splits['test']) if r['label']==1]
    original_clean_indices=[i for i,r in enumerate(splits['test']) if r['label']==0 and not r['derived_clean']]
    derived_indices=sorted([i for i,r in enumerate(splits['test']) if r['derived_clean']],
        key=lambda i:hashlib.sha256(('balanced20261027:'+splits['test'][i]['id']).encode()).hexdigest())
    needed=len(positive_indices)-len(original_clean_indices)
    if needed<0 or needed>len(derived_indices):
        raise RuntimeError('Cannot build balanced test while retaining all original test inputs')
    balanced_indices=positive_indices+original_clean_indices+derived_indices[:needed]
    balanced_rows=[splits['test'][i] for i in balanced_indices]
    balanced_metrics=metrics(balanced_rows,test_probs[balanced_indices],threshold)
    (OUT/'balanced_test_predictions.jsonl').write_text(''.join(json.dumps(dict(splits['test'][i],
        probability=float(test_probs[i]),route_to_refiner=bool(test_probs[i]>=threshold)),ensure_ascii=False)+'\n'
        for i in balanced_indices))
    original_by_id={r['id']:r for r in splits['test'] if not r['derived_clean']}
    probabilities={r['id']:float(p) for r,p in zip(splits['test'],test_probs)}
    original_test=[r for r in source if r['split']=='test']
    # Missing/ambiguous text labels are never used to bypass the Refiner.
    routes={r['id']:probabilities.get(r['id'],1.)>=threshold for r in original_test}
    lora_path=Path(os.environ.get('PREGATE_REFINER_PREDICTIONS',
        str(ROOT/'results/refiner13_qwen25_lora_20261026/evaluation/qwen25_lora.jsonl')))
    lora=[json.loads(s) for s in lora_path.open()]
    originals_by_id={r['id']:r for r in original_test}
    if {r['id'] for r in lora}!=set(originals_by_id):
        raise RuntimeError('Refiner predictions must match the same original test IDs')
    for r in lora:
        src=originals_by_id[r['id']]
        if r['input']!=src['input'] or r['reference']!=src['reference']:
            raise RuntimeError('Refiner predictions do not match test text: '+r['id'])
    gated=[]
    for r in lora:
        route=routes[r['id']]
        prediction=r['prediction'] if route else r['input']
        gated.append({'id':r['id'],'category':r['category'],'route_to_refiner':bool(route),
            'gate_probability':probabilities.get(r['id'],1.),'prediction':prediction,
            'reference':r['reference'],'input':r['input']})
    (OUT/'gated_lora_test.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in gated))
    identity=[r for r in gated if r['category']=='E07']
    summary={'status':'complete','parameter_count':parameter_count,'threshold_selected_on_dev':threshold,
        'validation':dev_metrics,'test_with_derived_clean_challenges':test_metrics,
        'balanced_50_50_gate_test':balanced_metrics,
        'balanced_50_50_note':'Detection only. Candidate clean references are weak labels. No combined Refiner quality or end-to-end latency claim for the balanced mix.',
        'refiner_prediction_file':str(lora_path),
        'original_1902_test':{'n':len(gated),'refiner_call_rate':sum(r['route_to_refiner'] for r in gated)/len(gated),
            'exact_match_rate':sum(r['prediction']==r['reference'] for r in gated)/len(gated),
            'E07_changed_count':sum(r['prediction']!=r['input'] for r in identity),'E07_n':len(identity),
            'need_refine_recall':sum(routes[r['id']] for r in original_test if r['input']!=r['reference'])/sum(r['input']!=r['reference'] for r in original_test)},
        'train_and_eval_elapsed_seconds':time.perf_counter()-started,'online_enabled':False,
        'label_status':'Weak paired-text supervision; not human gold. Existing test already inspected. No inference-speedup claim without latency measurement.'}
    if len(gated)!=1902:
        summary['original_test']=summary.pop('original_1902_test')
    tokenizer.save_pretrained(OUT/'tokenizer')
    model.encoder.config.save_pretrained(OUT/'encoder_config')
    write('summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    main()
