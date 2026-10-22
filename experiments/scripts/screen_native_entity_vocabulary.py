"""Development-only screen of the installed broad-vocabulary native backend.

Secondary diagnostic added because the primary native model cannot encode
about half of library entries. No held-out labels or audio are used here.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import sherpa_onnx

sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark_native_entity_streaming import read_jsonl,load_canonicals,hotword_payload,decode,sha256

MODEL=Path('/home/aim0/data/models/ASR/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16')


def main(out):
    tokens={s.rsplit(maxsplit=1)[0] for s in (MODEL/'tokens.txt').read_text().splitlines()}
    inputs=[r for r in read_jsonl(out/'inputs.jsonl') if r['split']=='dev']
    dest=out/'native_broad_vocab_development.jsonl'
    done={(r['id'],r['condition']) for r in read_jsonl(dest)}
    recognizers={}
    for condition in ['beam','hotword']:
        recognizers[condition]=sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(MODEL/'tokens.txt'),encoder=str(MODEL/'encoder-epoch-99-avg-1.int8.onnx'),
            decoder=str(MODEL/'decoder-epoch-99-avg-1.onnx'),joiner=str(MODEL/'joiner-epoch-99-avg-1.int8.onnx'),
            num_threads=2,enable_endpoint_detection=False,modeling_unit='cjkchar',provider='cpu',
            decoding_method='modified_beam_search',max_active_paths=4,hotwords_score=2.)
    for i,row in enumerate(inputs):
        words=load_canonicals(Path(row['library_path']),row['domain'])
        payload,unsupported=hotword_payload(words,tokens)
        assert sha256(Path(row['pcm_path']))==row['pcm_sha256']
        for condition,recognizer in recognizers.items():
            if (row['id'],condition) in done:continue
            result=decode(recognizer,row['pcm_path'],payload if condition=='hotword' else '')
            record=dict(id=row['id'],split='dev',condition=condition,score=2. if condition=='hotword' else None,
                        model=str(MODEL),requested_words=len(words),encoded_words=len(words)-len(unsupported),
                        unsupported_canonical=unsupported,pcm_sha256=row['pcm_sha256'],**result)
            with dest.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        if i%25==0:print('BROAD_VOCAB_DEV',i+1,'/',len(inputs),flush=True)
    print('BROAD_VOCAB_DEV_COMPLETE',len(inputs),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    main(p.parse_args().run_dir.resolve())
