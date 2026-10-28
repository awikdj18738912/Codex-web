"""Create a derived frozen LoRA dataset with every C02 pair excluded."""
from collections import Counter
import hashlib
import json
from pathlib import Path

SOURCE = Path('/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261026_provisional')
OUTPUT = Path('/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261027_no_C02')
BASE_CONFIG = Path(__file__).resolve().parents[1]/'llamafactory/refiner13_qwen25_lora_20261026_retry3.yaml'
NEW_CONFIG = Path(__file__).resolve().parents[1]/'llamafactory/refiner13_qwen25_lora_no_C02_20261027.yaml'
MODEL_OUTPUT = Path('/home/aim0/data/ASR/training/qwen25_05b_refiner_noC02_lora_20261027')
TOKENIZED = Path('/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261027_no_C02/tokenized_cache')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    if OUTPUT.exists() or MODEL_OUTPUT.exists() or NEW_CONFIG.exists():
        raise FileExistsError('Refusing to overwrite an existing no-C02 dataset/config/model output')
    source_manifest=load_json(SOURCE/'manifest.json')
    for name,expected in source_manifest['files_sha256'].items():
        if digest(SOURCE/name)!=expected:
            raise RuntimeError('Frozen source hash mismatch: '+name)
    sidecar=[json.loads(s) for s in (SOURCE/'split_audit.jsonl').open(encoding='utf-8')]
    sidecars={s:[r for r in sidecar if r['split']==s] for s in ('train','dev','test')}
    result={}
    kept_sidecar=[]
    excluded={}
    for split in ('train','dev','test'):
        convos=load_json(SOURCE/f'{split}.json')
        rows=sidecars[split]
        if len(convos)!=len(rows):
            raise RuntimeError(f'{split}: source conversations/sidecar misaligned')
        keep=[]; drop=[]
        for convo,meta in zip(convos,rows):
            msgs=convo['messages']
            user=next(m['content'] for m in msgs if m['role']=='user')
            assistant=next(m['content'] for m in msgs if m['role']=='assistant')
            if user!=meta['input'] or assistant!=meta['reference']:
                raise RuntimeError(f'{split}: source text mismatch at '+meta['id'])
            (drop if meta['category']=='C02' else keep).append((convo,meta))
        result[split]=[c for c,_ in keep]
        kept_sidecar.extend(m for _,m in keep)
        excluded[split]=[m for _,m in drop]
        print(split,'kept',len(keep),'removed_C02',len(drop),
            'class_counts',dict(Counter(m['category'] for _,m in keep)))

    OUTPUT.mkdir(parents=True)
    for split,rows in result.items():
        (OUTPUT/f'{split}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUTPUT/'split_audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept_sidecar),encoding='utf-8')
    info=load_json(SOURCE/'dataset_info.json')
    info={
        'refiner13_no_c02_train':dict(info['refiner13_train'],file_name='train.json'),
        'refiner13_no_c02_dev':dict(info['refiner13_dev'],file_name='dev.json'),
    }
    (OUTPUT/'dataset_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if any(r['category']=='C02' for r in kept_sidecar):
        raise RuntimeError('C02 row leaked into derived data')
    if len(kept_sidecar)!=sum(map(len,result.values())):
        raise RuntimeError('Derived sidecar row count mismatch')
    manifest={
        'status':'provisional_training_experiment_without_C02',
        'warning':'Source ledger remains pending human review; removing C02 does not certify remaining labels.',
        'source_dataset':str(SOURCE),
        'source_manifest_sha256':digest(SOURCE/'manifest.json'),
        'source_manifest_status':source_manifest['status'],
        'excluded_category':'C02',
        'excluded_counts':{s:len(v) for s,v in excluded.items()},
        'excluded_ids_by_split':{s:[r['id'] for r in rows] for s,rows in excluded.items()},
        'split_counts':{s:len(v) for s,v in result.items()},
        'class_counts':{s:dict(Counter(r['category'] for r in kept_sidecar if r['split']==s)) for s in result},
        'source_counts':{s:dict(Counter(r['source'] for r in kept_sidecar if r['split']==s)) for s in result},
        'system_prompt':source_manifest['system_prompt'],
        'files_sha256':{name:digest(OUTPUT/name) for name in ('train.json','dev.json','test.json','split_audit.jsonl','dataset_info.json')},
    }
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    config=BASE_CONFIG.read_text(encoding='utf-8')
    config=config.replace('# Same frozen provisional data and LoRA settings as first two attempts.',
        '# Same LoRA recipe; derived train/dev/test exclude all C02 examples.')
    config=config.replace('dataset_dir: /home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261026_provisional',
        f'dataset_dir: {OUTPUT}')
    config=config.replace('dataset: refiner13_train','dataset: refiner13_no_c02_train')
    config=config.replace('eval_dataset: refiner13_dev','eval_dataset: refiner13_no_c02_dev')
    config=config.replace('/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261026_provisional/tokenized_cache',str(TOKENIZED))
    config=config.replace('/home/aim0/data/ASR/training/qwen25_05b_refiner13_lora_20261026_retry3',str(MODEL_OUTPUT))
    NEW_CONFIG.write_text(config,encoding='utf-8')
    print('manifest',json.dumps(manifest,ensure_ascii=False))
    print('config',NEW_CONFIG)


if __name__=='__main__': main()
