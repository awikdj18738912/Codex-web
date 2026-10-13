"""Materialize the independently fixed list, without looking at ASR outputs."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import shutil
import sys
import wave

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c


def prepare(root):
    import pyarrow.parquet as pq
    import numpy as np
    plan=json.loads((root/'source_candidates.json').read_text());bank=root/'natural_holdout'
    if bank.exists():raise FileExistsError('Do not overwrite a prepared natural holdout')
    source_model=ROOT/'results/gate_linear_holdout_20261012';bank.mkdir();(bank/'logs').mkdir();(bank/'audio').mkdir();(bank/'executed_code').mkdir()
    shutil.copytree(root/'runtime',bank/'runtime');shutil.copytree(source_model/'frozen_model',bank/'frozen_model')
    shutil.copy2(source_model/'selection.json',bank/'selection.json');parquets={};rows=[]
    for item in plan['candidates']:
        path=Path(item['parquet_absolute_path'])
        if str(path) not in parquets:parquets[str(path)]=pq.read_table(path)
        row=parquets[str(path)].slice(item['row_index'],1).to_pylist()[0]
        assert row['segment_id']==item['segment_id'] and row['text']==item['reference']
        payload=row['audio']['bytes']
        with wave.open(io.BytesIO(payload),'rb') as wav:
            assert (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())==(16000,1,2)
            pcm=wav.readframes(wav.getnframes());duration=wav.getnframes()/16000
        output=bank/'audio'/(c.digest(item['segment_id'])[:20]+'.wav');output.write_bytes(payload)
        floatpcm=(np.frombuffer(pcm,dtype='<i2').astype('<f4')/32768.).astype('<f4').tobytes()
        rows.append(dict(seed_id='speechio_holdout:'+item['segment_id'],category='natural_speech',data_kind='human_spontaneous',
            template_group='speechio_shard:'+item['source_shard'],speaker=item['speaker'],source_id=item['segment_id'],
            source_shard=item['source_shard'],source_parquet=str(path),source_row=item['row_index'],
            audio_path=str(output),audio_sha256=c.file_hash(output),pcm_sha256=c.digest(floatpcm),duration_seconds=duration,
            reference=row['text'],intended_text=row['text'],spoken_text=row['text'],domain='general',human_gold=True,
            reference_kind='SpeechIO_dataset_verbatim_reference_not_cleaned_target',audio_script_verified=False,
            source_text_prior_resource_exposure_unknown=True))
    for row in rows:c.append_row(bank/'human_sources.jsonl',row)
    test=bank/'test_data';test.mkdir();(test/'tts.jsonl').write_text('')
    c.write_json(test/'manifest.json',dict(count=0,no_synthesis_needed=True))
    manifest=dict(synthetic_audio=0,human_audio=len(rows),human_sources_sha256=c.file_hash(bank/'human_sources.jsonl'),
        selection_sha256=c.file_hash(bank/'selection.json'),model_sha256=c.file_hash(bank/'frozen_model/model.pkl'),
        threshold=.8,runtime_protocol_sha256=c.file_hash(bank/'runtime/protocol.json'),
        streaming_adapter_sha256=c.file_hash(ROOT/'experiments/scripts/stream_linear_gate_holdout.py'),
        selection_plan_sha256=c.file_hash(root/'source_candidates.json'),
        source_parquets_sha256={path:c.file_hash(path) for path in parquets},
        generator_sha256=c.file_hash(Path(__file__)),scope='new_natural_speech_project_unused_ID_list_with_verbatim_refs',
        reference_caveat='Reference preserves fillers/repetitions; CER assesses fidelity, not cleanup benefit',
        ASR_pretraining_overlap_unknown=True,threshold_tuning_prohibited=True)
    c.write_json(bank/'manifest.json',manifest);(bank/'executed_code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('Prepared natural clips',len(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();prepare(a.root.resolve())
