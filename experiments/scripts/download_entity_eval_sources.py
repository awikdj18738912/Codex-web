"""Download fixed SpeechIO evaluation assets and verify repository LFS hashes."""
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/entity_reliable_data_20261012/source/speechio'
REPO = 'yuekai/speechio_test'
REVISION = 'bdba6aa3daeddb2996eaa31c44bbdf633a07a31f'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(repository=REPO, revision=REVISION, purpose='evaluation_only',
                    original_source='https://github.com/SpeechColab/Leaderboard',
                    mirror_license='apache-2.0 (dataset card; not a blanket license for source programmes)',
                    assets=[], errors=[])
    for subset in ('SPEECHIO_ASR_ZH00003', 'SPEECHIO_ASR_ZH00021'):
        try:
            metadata = requests.get(f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{subset}',
                                    params={'expand':'true'}, timeout=25)
            metadata.raise_for_status()
            item = next(x for x in metadata.json() if x['path'].endswith('.parquet'))
            destination = OUT / f'{subset}.parquet'
            expected = item['lfs']['oid']
            if destination.exists():
                assert destination.stat().st_size == item['size']
                assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected
            else:
                for attempt in range(3):
                    partial = destination.with_suffix('.parquet.partial')
                    try:
                        response = requests.get(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{item["path"]}',
                                                params={'download':'true','entity_eval':f'20261012-{int(time.time())}'},
                                                stream=True, timeout=(15,30))
                        response.raise_for_status()
                        digest = hashlib.sha256()
                        count = 0
                        with partial.open('wb') as f:
                            for block in response.iter_content(1024*1024):
                                if block:
                                    f.write(block); digest.update(block); count += len(block)
                        response.close()
                        assert count == item['size'], (count,item['size'])
                        assert digest.hexdigest() == expected
                        partial.replace(destination)
                        break
                    except Exception as error:
                        print(subset, 'attempt', attempt+1, type(error).__name__, flush=True)
                        if attempt == 2:
                            raise
            manifest['assets'].append(dict(subset=subset, path=str(destination), bytes=item['size'],
                                           sha256=expected, remote_path=item['path'], verified=True))
            print('VERIFIED', subset, item['size'], flush=True)
        except Exception as error:
            manifest['errors'].append(dict(subset=subset, error_type=type(error).__name__))
        (OUT / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    if manifest['errors']:
        raise RuntimeError('Some source assets failed; see manifest without signed URL details')


if __name__ == '__main__':
    main()
