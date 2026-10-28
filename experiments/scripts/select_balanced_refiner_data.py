"""Produce auditable, capped candidate pools; never export training-ready labels.

Public native annotations, model audits and unresolved semantic checks are kept
distinct. The selected files remain pending human review; missing classes stay
missing instead of being padded with duplicates or invented answers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

from fcgec_public_adapter import iter_fcgec_candidates
from generate_refiner_100k import NearIndex, QUOTAS, construct_validate
from generate_gemma4_selfgen_pilot import audit_pass, content


CATEGORIES = list(QUOTAS)
NAMES = dict(zip(CATEGORIES, ['填充词', '口吃', '重复', '重启', '单次改口', '多次改口',
    '拼字说明', '错字错词', '语法赘余', '缺词', '语序', '标点', '异常断句',
    '否定条件立场', '英文型号', '占位符', '正确句']))
ASCII = re.compile(r'[A-Za-z][A-Za-z0-9_+./-]*')
NUMBERS = re.compile(r'\d+(?:[.,:/%-]\d+)*')
ENTITY = re.compile(r'__ENTITY_\d{3}__')
NEGATION = re.compile(r'不|没|未|无|非|只有|除非|只要')


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(value):
    # Key only: original sample bytes are never rewritten by normalization.
    return ' '.join(unicodedata.normalize('NFKC', value).split())


def lexical(value):
    return content(normalized(value))


def dedup_text(value):
    mapping = {}
    def rename(match):
        tag = match.group()
        if tag not in mapping:
            mapping[tag] = f'__ENTITY_{len(mapping):03d}__'
        return mapping[tag]
    return ENTITY.sub(rename, value)


def edits(source, target):
    return [(op, source[a:b], target[c:d], a, b, c, d)
            for op, a, b, c, d in SequenceMatcher(None, source, target, autojunk=False).get_opcodes()
            if op != 'equal']


def common_check(row):
    source, target = row.get('input'), row.get('reference')
    if not all(isinstance(t, str) and t.strip() and 4 <= len(t) <= 320 for t in (source, target)):
        return 'empty_or_length_outside_4_320'
    if any(marker in source+target for marker in ('```', '<think>', '</think>', '\x00')):
        return 'markup_or_control_token'
    if ENTITY.findall(source) != ENTITY.findall(target):
        return 'placeholder_sequence_changed'
    if row['category'] not in ('B01', 'B02'):
        if NUMBERS.findall(source) != NUMBERS.findall(target):
            return 'number_sequence_changed'
        if ASCII.findall(source) != ASCII.findall(target):
            return 'ascii_sequence_changed'
    return None


def record(dataset, identifier, category, source, target, *, group=None,
           eligible=False, reason=None, basis=None, **extra):
    return {'id': f'{dataset}:{identifier}', 'dataset': dataset,
            'source_id': str(identifier), 'group_id': group or f'{dataset}:{identifier}',
            'original_split': 'train', 'category': category,
            'input': source, 'reference': target, 'eligible': eligible,
            'review_reason': reason, 'quality_basis': basis,
            'human_review_status': 'pending', 'training_ready': False, **extra}


def iter_synthetic(snapshot):
    with sqlite3.connect(f'file:{snapshot}?mode=ro', uri=True) as db:
        for identifier, category, source, target, raw in db.execute(
                "SELECT id,category,source,target,payload_json FROM samples WHERE status='accepted'"):
            payload = json.loads(raw)
            reason = None
            try:
                if not audit_pass(payload.get('audit', {}), target):
                    raise ValueError('saved_model_audit_does_not_pass')
                rebuilt = construct_validate(payload['raw_teacher_sample'], category, payload['plan'])
                if rebuilt['input'] != source or rebuilt['reference'] != target:
                    raise ValueError('reconstruction_differs_from_saved_pair')
            except (ValueError, KeyError, TypeError) as error:
                reason = str(error)
            yield record('Gemma4', identifier, category, source, target,
                         group='Gemma4:'+payload['group_id'], eligible=reason is None,
                         reason=reason, basis='saved_model_audit_and_program_revalidation',
                         original_config_hash=payload.get('config_hash'),
                         source_ledger_snapshot=str(snapshot),
                         audit=payload.get('audit'), scene=payload.get('plan', {}).get('scene'),
                         semantic_review='saved_teacher_model_pass',
                         auxiliary_labels=protection_labels(source, target))


def protection_labels(source, target):
    labels = []
    if NEGATION.search(source) and NEGATION.findall(source) == NEGATION.findall(target):
        labels.append('E03')
    if ASCII.findall(source) and ASCII.findall(source) == ASCII.findall(target):
        labels.append('E05')
    if ENTITY.findall(source) and ENTITY.findall(source) == ENTITY.findall(target):
        labels.append('E06')
    return labels


def public_label_evidence(row):
    """Require an explicit original label matching the project's main class.

    Edit patterns, homophones, model guesses and ASR reference pairs do not
    establish a native fine-grained class. This gate supplements quality checks.
    """
    dataset, category = row['dataset'], row['category']
    if dataset == 'Gemma4':
        return {'label_origin': 'generated_plan_and_saved_model_audit', 'category': category}
    native = row.get('native_labels')
    if dataset == 'CS2W' and isinstance(native, dict):
        mapping = {
            'A01': ('disfluency_label', '1', 'filler_words'),
            'C02': ('grammatical_error_label', '1', 'redundant_words'),
            'C03': ('grammatical_error_label', '2', 'missing_words'),
        }
        if category in mapping:
            field, code, meaning = mapping[category]
            if code in str(native.get(field, '')):
                return {'label_origin': 'official_native_annotation', 'field': field,
                        'code': code, 'native_meaning': meaning, 'category': category}
    if dataset == 'FCGEC' and isinstance(native, str):
        mapping = {'C02': 'CR', 'C03': 'CM', 'C04': 'IWO', 'E07': '*'}
        if mapping.get(category) == native:
            return {'label_origin': 'official_native_annotation', 'field': 'error_type',
                    'code': native, 'category': category,
                    'note': 'E07 also requires consistent error_flag=0 in the adapter' if category == 'E07' else ''}
    return None


def enforce_public_label_policy(row):
    evidence = public_label_evidence(row)
    row['category_label_evidence'] = evidence
    if row['dataset'] == 'Gemma4':
        return
    # Keyword detections are search hints, never accepted auxiliary labels.
    row['inferred_protection_signals'] = row.pop('auxiliary_labels', [])
    row['auxiliary_labels'] = []
    if evidence is None:
        if row['category'] != 'unclassified':
            row['suggested_category_not_approved'] = row['category']
        row['prior_review_reason'] = row.get('review_reason')
        row.update(category='unclassified', eligible=False,
                   review_reason='no_explicit_matching_native_subclass_label',
                   category_label_status='unlabeled_for_project_subclass')
    else:
        row['category_label_status'] = 'explicit_native_mapping'


def cs2w_classify(row):
    source, target = row['content'], row['ref']
    fields = ('disfluency_label', 'grammatical_error_label', 'colloquial_word_label', 'asr_error_label')
    if any(not isinstance(row.get(f), str) or len(row[f]) != len(source) for f in fields):
        return 'unclassified', False, 'native_label_length_mismatch'
    active = {f: set(row[f]) - {'0'} for f in fields}
    pure_disfluency = active['disfluency_label'] and all(not active[f] for f in fields[1:])
    pure_grammar = active['grammatical_error_label'] and all(not active[f] for f in (fields[0], fields[2], fields[3]))
    a, b = lexical(source), lexical(target)
    diff = edits(a, b)
    if pure_disfluency and active[fields[0]] == {'1'}:
        category = 'A01'
    elif pure_disfluency and active[fields[0]] == {'2'}:
        category = 'A04'  # A tentative queue label, not an approved subtype.
        if len(diff) == 1 and diff[0][0] == 'delete':
            span = diff[0][1]
            if len(span) == 1 and span*2 in a:
                category = 'A02'
            elif 2 <= len(span) <= 8 and span*2 in a:
                category = 'A03'
    elif pure_grammar and active[fields[1]] == {'1'}:
        category = 'C02'
    elif pure_grammar and active[fields[1]] == {'2'}:
        return 'C03', False, 'native_missing_word_requires_semantic_review'
    elif active[fields[3]]:
        return 'C01', False, 'mixed_or_asr_pair_requires_semantic_review'
    else:
        return 'unclassified', False, 'mixed_native_labels_or_out_of_scope_rewrite'
    if len(diff) != 1 or diff[0][0] != 'delete':
        return category, False, 'not_one_local_deletion'
    _, span, _, start, end, _, _ = diff[0]
    field, code = (fields[1], '1') if category == 'C02' else (fields[0], '1' if category == 'A01' else '2')
    lexical_labels = [row[field][i] for i, c in enumerate(source)
                      if not c.isspace() and unicodedata.category(c)[0] not in 'PS']
    # Avoid normalization-induced index drift; such cases get manual review.
    if len(lexical_labels) != len(a) or any(v != code for v in lexical_labels[start:end]):
        return category, False, 'deleted_span_not_fully_supported_by_native_labels'
    if category == 'A01' and span not in {'呃', '嗯', '额'}:
        return category, False, 'filler_has_possible_lexical_meaning'
    if category == 'A02' and span not in set('我你他她它这那今明请先但可因就所为'):
        return category, False, 'single_character_repetition_needs_subtype_review'
    if category == 'A03':
        return category, False, 'repetition_vs_emphasis_requires_semantic_review'
    if category == 'A04':
        return category, False, 'restart_subtype_requires_semantic_classification'
    if category == 'C02' and (len(span) > 8 or NEGATION.search(span)):
        return category, False, 'redundancy_span_requires_semantic_review'
    return category, True, None


def iter_cs2w(path, revision):
    for line_number, line in enumerate(path.open(), 1):
        raw = json.loads(line)
        category, eligible, reason = cs2w_classify(raw)
        yield record('CS2W', f'{raw["id"]}:line{line_number}', category, raw['content'], raw['ref'],
                     eligible=eligible, reason=reason,
                     basis='official_native_annotation_and_local_edit_checks',
                     revision=revision, native_id=raw['id'], source_line=line_number,
                     semantic_review='native_annotation_only',
                     native_labels={k: v for k, v in raw.items() if k.endswith('_label')},
                     auxiliary_labels=protection_labels(raw['content'], raw['ref']))


def fcgec_gate(raw):
    category = raw['category']
    if not raw['valid']:
        return False, raw['diagnostic']
    if category == 'pending':
        return False, 'mixed_or_out_of_scope_native_error_type'
    source, target = raw['input'], raw['reference']
    a, b = lexical(source), lexical(target)
    diff = edits(a, b)
    if category == 'E07':
        if source != target:
            return False, 'native_correct_sentence_has_edits'
        if re.search(r'呃|嗯|额', source):
            return False, 'native_written_correctness_not_clean_oral_guarantee'
        return True, None
    if NEGATION.findall(source) != NEGATION.findall(target):
        return False, 'negation_changed_requires_semantic_review'
    if category == 'C02':
        ok = len(diff) == 1 and diff[0][0] == 'delete' and 1 <= len(diff[0][1]) <= 8
        return ok, None if ok else 'not_one_local_grammar_deletion'
    if category == 'C03':
        return False, 'missing_word_requires_semantic_review'
    if category == 'C04':
        if Counter(a) != Counter(b) or a == b:
            return False, 'word_order_includes_lexical_changes'
        prefix = 0
        while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
            prefix += 1
        suffix = 0
        while suffix < min(len(a), len(b))-prefix and a[-suffix-1] == b[-suffix-1]:
            suffix += 1
        changed = len(a)-prefix-suffix
        ok = 2 <= changed <= 16
        return ok, None if ok else 'word_order_not_local_within_16_characters'
    return False, 'unsupported_category'


def iter_fcgec(path, revision):
    # Multi-reference examples count once. Prefer a valid, eligible, minimally
    # edited native reference and retain all alternatives for later review.
    grouped = defaultdict(list)
    for raw in iter_fcgec_candidates(path, revision, 'train_filtered'):
        grouped[raw['group_id']].append(raw)
    for group, refs in grouped.items():
        ranked = []
        for raw in refs:
            eligible, reason = fcgec_gate(raw)
            distance = sum(max(len(x[1]), len(x[2])) for x in edits(raw['input'], raw['reference'])) if raw['reference'] is not None else 100000
            ranked.append((not eligible, not raw['valid'], distance, raw['reference_index'], raw, reason))
        _, _, _, _, raw, reason = min(ranked, key=lambda x: x[:4])
        yield record('FCGEC', group, raw['category'] if raw['category'] != 'pending' else 'unclassified',
                     raw['input'], raw['reference'], eligible=fcgec_gate(raw)[0],
                     reason=reason, basis='official_native_annotation_and_verified_edit_reconstruction',
                     revision=revision, semantic_review='native_annotation_only',
                     native_labels=raw['native_labels'], operation=raw['operation'],
                     reference_index=raw['reference_index'],
                     alternative_references=[{'reference': r['reference'], 'operation': r['operation'],
                         'valid': r['valid'], 'diagnostic': r['diagnostic']} for r in refs],
                     auxiliary_labels=protection_labels(raw['input'], raw['reference'] or ''))


def iter_chinesehp(path, revision):
    for line in path.open():
        raw = json.loads(line)
        source, target = raw['nbest'][0].strip(), raw['reference'].strip()
        # ChineseHP supplies ASR hypotheses/references, not these 17 native
        # class labels. Keep the material, but never infer admission from text.
        yield record('ChineseHP', raw['id'], 'unclassified', source, target,
                     group='ChineseHP:'+raw['corpus']+':'+raw['id'],
                     eligible=False, reason='no_explicit_matching_native_subclass_label',
                     basis='official_asr_pair_fixed_top1_without_project_subclass_label',
                     revision=revision, corpus=raw['corpus'], nbest_index=0,
                     candidate_selection='fixed_first_hypothesis_without_reference_oracle',
                     semantic_review='not_admitted_without_explicit_subclass_label',
                     auxiliary_labels=protection_labels(source, target))


def heldout_texts(public_root, test_path):
    for split in ('val', 'test'):
        for line_number, line in enumerate((public_root/'CS2W'/f'{split}.jsonl').open(), 1):
            raw = json.loads(line)
            yield f'CS2W/{split}/{raw["id"]}/line{line_number}/source', raw['content']
            yield f'CS2W/{split}/{raw["id"]}/line{line_number}/target', raw['ref']
    for path, split in ((public_root/'FCGEC/FCGEC_valid.json', 'valid'), (test_path, 'test')):
        if path is None or not path.exists():
            continue
        data = json.loads(path.read_text())
        for identifier, raw in data.items():
            yield f'FCGEC/{split}/{identifier}/source', raw['sentence']
        if split == 'valid':
            for raw in iter_fcgec_candidates(path, 'heldout', split):
                if raw['valid']:
                    yield f'FCGEC/{split}/{raw["id"]}/target', raw['reference']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--public-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-per-class', type=int, default=5882)
    parser.add_argument('--fcgec-test', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists():
        raise ValueError('completed_output_exists_use_a_new_run_directory')
    manifest = json.loads((args.public_root/'manifest.json').read_text())
    revs = {name: value['revision'] for name, value in manifest['retrieval_notes']['repositories'].items()}
    hp_path = args.public_root/'ChineseHP/paper_train_selected.jsonl'
    hp_revision = '8bbea809578d81f4854b58c9073506349dbf1d7d'
    # Manifest its actual bytes and all code used; do not rely on mutable names.
    input_paths = [args.snapshot, args.public_root/'manifest.json',
                   args.public_root/'CS2W/train.jsonl', args.public_root/'FCGEC/FCGEC_train_filtered.json']
    if hp_path.exists():
        input_paths += [hp_path, args.public_root/'asr_text_sources_manifest.json']
    if args.fcgec_test and args.fcgec_test.exists():
        input_paths.append(args.fcgec_test)
    def file_hash(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
        return h.hexdigest()
    cfg = {'created_utc': datetime.now(timezone.utc).isoformat(), 'target_per_class': args.target_per_class,
           'target_total': args.target_per_class*len(CATEGORIES), 'categories': CATEGORIES,
           'inputs': {str(p.resolve()): file_hash(p) for p in input_paths},
           'selection_seed': 'balanced-20261025-v1', 'near_duplicate_jaccard': .85,
           'public_native_annotations_and_model_audits_are_distinct': True,
           'public_admission_policy': 'explicit_matching_original_subclass_label_required',
           'public_unlabeled_material_must_not_fill_any_class_quota': True,
           'human_review_required': True, 'training_started': False,
           'heldout_coverage': ['CS2W val/test source+reference', 'FCGEC valid source+references',
               'FCGEC test source' if args.fcgec_test and args.fcgec_test.exists() else 'FCGEC test unavailable; native filtered train only'],
           'chft_train_excluded_reason': 'Available train schema is key/ref only, no paired hypothesis; partial download excluded.'}
    code_dir = out/'code_snapshot'; code_dir.mkdir(exist_ok=True)
    for path in (Path(__file__), Path(__file__).with_name('fcgec_public_adapter.py'),
                 Path(__file__).with_name('generate_refiner_100k.py'),
                 Path(__file__).with_name('generate_gemma4_selfgen_pilot.py')):
        (code_dir/path.name).write_bytes(path.read_bytes())
    (out/'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+'\n')
    db = sqlite3.connect(out/'selection.sqlite3')
    db.executescript('''CREATE TABLE IF NOT EXISTS candidates(
        id TEXT PRIMARY KEY, category TEXT, dataset TEXT, source_key TEXT, target_key TEXT,
        group_key TEXT, eligible INTEGER, state TEXT, reason TEXT, rank TEXT, payload TEXT);
        CREATE INDEX IF NOT EXISTS source_lookup ON candidates(source_key);
        CREATE INDEX IF NOT EXISTS category_state ON candidates(category,state);''')
    def ingest(iterator):
        total = 0
        for row in iterator:
            enforce_public_label_policy(row)
            if row['eligible']:
                failure = common_check(row)
                if failure:
                    row.update(eligible=False, review_reason=failure)
            source, target = row.get('input') or '', row.get('reference') or ''
            row['supplemental_labels'] = row.pop('auxiliary_labels', [])
            rank = ('1' if row['dataset'] == 'Gemma4' else '0') + digest(cfg['selection_seed']+row['id'])
            db.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                       (row['id'], row['category'], row['dataset'], normalized(source), normalized(target),
                        row['group_id'], int(row['eligible']), 'eligible' if row['eligible'] else 'needs_review',
                        row.get('review_reason'), rank, json.dumps(row, ensure_ascii=False)))
            total += 1
        db.commit()
        print('INGESTED', total, flush=True)
    ingest(iter_synthetic(args.snapshot.resolve()))
    ingest(iter_cs2w(args.public_root/'CS2W/train.jsonl', revs['CS2W']))
    ingest(iter_fcgec(args.public_root/'FCGEC/FCGEC_train_filtered.json', revs['FCGEC']))
    if hp_path.exists():
        ingest(iter_chinesehp(hp_path, hp_revision))
    # Incompatible targets from independent sources are never silently resolved.
    conflict_sources = db.execute("SELECT source_key FROM candidates WHERE eligible=1 GROUP BY source_key HAVING COUNT(DISTINCT target_key)>1").fetchall()
    for (source,) in conflict_sources:
        db.execute("UPDATE candidates SET state='quarantined',reason='same_source_conflicting_targets' WHERE source_key=? AND eligible=1", (source,))
    db.commit()
    heldout = NearIndex(); heldout_exact = set()
    for identifier, text in heldout_texts(args.public_root, args.fcgec_test):
        heldout_exact.add(normalized(text)); heldout.add(identifier, dedup_text(text))
    print('HELDOUT_INDEX', len(heldout.texts), flush=True)
    selected_count = Counter(); selected_sources = set(); selected_groups = set(); near = NearIndex()
    rows = db.execute("SELECT id,category,source_key,group_key,payload FROM candidates WHERE state='eligible' ORDER BY rank,id").fetchall()
    for i, (identifier, category, source_key, group, payload) in enumerate(rows):
        row = json.loads(payload)
        source, target = row['input'], row['reference']
        reason = None
        if normalized(source) in heldout_exact or normalized(target) in heldout_exact:
            reason = 'exact_overlap_with_official_heldout'
        else:
            match = heldout.match(dedup_text(source)) or heldout.match(dedup_text(target))
            if match:
                reason = 'lexical_near_overlap_with_official_heldout:'+match
        if reason:
            state = 'heldout_overlap'
        elif source_key in selected_sources or group in selected_groups:
            state, reason = 'duplicate', 'same_input_or_source_group_already_selected'
        else:
            match = near.match(dedup_text(source)) or near.match(dedup_text(target))
            if match:
                state, reason = 'near_duplicate', 'lexical_near_duplicate_of_selected:'+match
            elif selected_count[category] >= args.target_per_class:
                state, reason = 'reserve', 'class_quota_excess'
            else:
                state = 'selected_pending_human_review'
                selected_count[category] += 1
                selected_sources.add(source_key); selected_groups.add(group)
                near.add(identifier+'/input', dedup_text(source))
                near.add(identifier+'/target', dedup_text(target))
        db.execute('UPDATE candidates SET state=?,reason=? WHERE id=?', (state, reason, identifier))
        if i % 2000 == 0:
            db.commit(); print('SCREENED', i, dict(selected_count), flush=True)
    db.commit()
    byclass = out/'by_category'; byclass.mkdir(exist_ok=True)
    for name, condition in [('selected.pending_review.jsonl', "state='selected_pending_human_review'"),
                            ('reserve.jsonl', "state='reserve'"),
                            ('public_needs_review.jsonl', "state='needs_review' AND dataset!='Gemma4'"),
                            ('synthetic_needs_review.jsonl', "state='needs_review' AND dataset='Gemma4'"),
                            ('excluded.jsonl', "state NOT IN ('selected_pending_human_review','reserve','needs_review')")]:
        with (out/name).open('w') as stream:
            for payload, state, reason in db.execute('SELECT payload,state,reason FROM candidates WHERE '+condition+' ORDER BY category,rank,id'):
                row = json.loads(payload); row.update(selection_state=state, selection_reason=reason)
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'target_per_class': args.target_per_class,
               'target_total': args.target_per_class*17, 'selected_total': sum(selected_count.values()),
               'human_reviewed_total': 0, 'training_ready': False,
               'status_counts': dict(db.execute('SELECT state,count(*) FROM candidates GROUP BY state')),
               'by_category': {}, 'selected_by_source': dict(db.execute("SELECT dataset,count(*) FROM candidates WHERE state='selected_pending_human_review' GROUP BY dataset"))}
    review = ['# 17类均衡候选：分组抽样审核', '', '> 所有样本仍待人工审核。每组展示至多5条；请在“修改意见”列记录。', '']
    for category in CATEGORIES:
        stats = {state: n for state, n in db.execute('SELECT state,count(*) FROM candidates WHERE category=? GROUP BY state', (category,))}
        selected = selected_count[category]
        sources = dict(db.execute("SELECT dataset,count(*) FROM candidates WHERE category=? AND state='selected_pending_human_review' GROUP BY dataset", (category,)))
        summary['by_category'][category] = {'name': NAMES[category], 'target': args.target_per_class,
            'selected': selected, 'shortfall': args.target_per_class-selected,
            'reserve': stats.get('reserve', 0), 'pending_semantic_or_subtype_review': stats.get('needs_review', 0),
            'selected_by_source': sources, 'states': stats}
        with (byclass/f'{category}.pending_review.jsonl').open('w') as stream:
            for payload in db.execute("SELECT payload FROM candidates WHERE category=? AND state='selected_pending_human_review' ORDER BY rank,id", (category,)):
                row = json.loads(payload[0]); row['selection_state']='selected_pending_human_review'
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        review += [f'## {category} {NAMES[category]}', '', '| ID／来源 | 输入 | 候选目标 | 审核依据 | 修改意见 |', '| --- | --- | --- | --- | --- |']
        for (payload,) in db.execute("SELECT payload FROM candidates WHERE category=? AND state='selected_pending_human_review' ORDER BY rank,id LIMIT 5", (category,)):
            row = json.loads(payload)
            esc = lambda value: str(value).replace('|', '／').replace('\n', ' ')
            review.append('| '+' | '.join(map(esc, [row['id'], row['input'], row['reference'], row['quality_basis'], '待人工复核']))+' |')
        review.append('')
    (out/'抽样审核.md').write_text('\n'.join(review))
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    report = ['# 每类5882条：均衡筛选结果', '',
              f'目标：17×{args.target_per_class}={args.target_per_class*17}条。本次暂选{summary["selected_total"]}条，缺口{args.target_per_class*17-summary["selected_total"]}条。', '',
              '**这些是按已存模型审核或原生标注＋程序检查筛出的候选，仍待人工复核；不是全部经过模型二审，更不是人工验收训练集。**', '',
              '| 小类 | 名称 | 目标 | 暂选去重候选 | 缺口 | 超额备用 | 待语义／细类审核 | 来源 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for category, stats in summary['by_category'].items():
        sources = '、'.join(f'{key} {value}' for key,value in stats['selected_by_source'].items())
        report.append(f'| {category} | {stats["name"]} | {args.target_per_class} | {stats["selected"]} | {stats["shortfall"]} | {stats["reserve"]} | {stats["pending_semantic_or_subtype_review"]} | {sources} |')
    report += ['', '公开准入：必须有明确对应的原生类别标签。CS2W R-type不能直接映射A02/A03/A04；ChineseHP无本项目小类原生标签，只作未标注素材，不进入任何小类配额。关键词／拼音／编辑规则不能代替标签。',
               '筛选口径：有明确标签的公开来源优先、每个源组只计一次；输入冲突隔离；官方验证测试优先保留；做精确与词面近重复筛查（MinHash候选检索＋Jaccard≥0.85确认），不声称语义去重保证。',
               'E06去重键统一占位符编号，但原始输入目标保持原样。多参考保存在元数据，同源只选一条，不跨划分。',
               'C03缺词及不明确的CS2W细类放待审池，不为凑配额强行批准。已有超量候选保留备用，原始生成账本与公开源文件均未改动。',
               'ChFT当前训练发布物只有key/ref，未将不完整下载或单侧参考冒充纠错配对。官方标注与同模型审核均可能有误，须按来源／类别抽样检查误收与误拒。',
               '尚未执行最终训练token长度验收，也未导出SFT或启动训练。', '',
               '文件：`selected.pending_review.jsonl`、`by_category/`、`reserve.jsonl`、`public_needs_review.jsonl`、`excluded.jsonl`、`selection.sqlite3`、`抽样审核.md`、`summary.json`、`config.json`。', '']
    (out/'筛选统计.md').write_text('\n'.join(report))
    db.close()
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
