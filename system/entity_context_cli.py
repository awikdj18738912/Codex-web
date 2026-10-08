"""Manage optional spelling contexts without changing existing Web endpoints."""
import argparse
import json
from pathlib import Path
from .entity_store import EntityStore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,required=True)
    sub=parser.add_subparsers(dest='action',required=True)
    enable=sub.add_parser('enable',help='Enable context-constrained spelling for a registered name')
    enable.add_argument('--id',type=int,required=True)
    enable.add_argument('--keyword',action='append',required=True,help='Independent context term; repeat for additional terms')
    enable.add_argument('--minimum-score',type=float,default=1.0)
    enable.add_argument('--source',required=True,help='Verified standard-name list or document provenance')
    enable.add_argument('--allow-internal-omission',action='store_true',help='Opt in to bounded internal-character completion for long terms')
    enable.add_argument('--allow-near-spelling',action='store_true',help='Opt in to guarded single-character near-sound correction for long medical terms')
    disable=sub.add_parser('disable');disable.add_argument('--id',type=int,required=True)
    sub.add_parser('list')
    export=sub.add_parser('export');export.add_argument('--output',type=Path,required=True)
    load=sub.add_parser('import');load.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();store=EntityStore(args.db)
    if args.action=='enable':
        entity=store.set_spelling_context(args.id,keywords={k:1.0 for k in args.keyword},minimum_score=args.minimum_score,source=args.source,
                                         allow_internal_omission=args.allow_internal_omission,allow_near_spelling=args.allow_near_spelling)
        print(json.dumps(dict(entity_id=entity.entity_id,canonical_text=entity.canonical_text,enabled=True),ensure_ascii=False))
    elif args.action=='disable':
        print(json.dumps(dict(entity_id=args.id,disabled=store.clear_spelling_context(args.id))))
    elif args.action=='import':
        records=json.loads(args.input.read_text())
        if not isinstance(records,list):raise ValueError('Expected a JSON array of standard-name records')
        # Validate an import against a temporary database first, so malformed
        # files cannot leave a partly configured live glossary.
        import sqlite3,tempfile
        def contents(connection):
            return {
                table:sorted(tuple(row) for row in connection.execute('SELECT * FROM '+table))
                for table in ['protected_entities','entity_aliases','entity_spelling_contexts']
            }
        with tempfile.TemporaryDirectory() as folder:
            temporary=Path(folder)/'import.db'
            with sqlite3.connect(args.db) as src,sqlite3.connect(temporary) as dst:src.backup(dst)
            staged=EntityStore(temporary)
            with staged._connect() as connection:
                original=contents(connection)
            def apply(target):
                for r in records:
                    if 'enabled' in r and not isinstance(r['enabled'],bool):
                        raise ValueError('enabled must be a boolean')
                    try:existing=target.get_entity(r['canonical_text'],r.get('domain','general'))
                    except KeyError:existing=None
                    d=target.upsert_entity(r['canonical_text'],entity_type=r.get('entity_type',existing.entity_type if existing else 'TERM'),
                        domain=r.get('domain','general'),normalization_policy=r.get('normalization_policy',existing.normalization_policy if existing else 'normalize'),
                        priority=r.get('priority',existing.priority if existing else 0),
                        aliases=r.get('aliases',existing.aliases if existing else ()),source=r['source'])
                    if not r.get('enabled',existing.enabled if existing else True):
                        target.set_enabled_by_id(d.entity_id,enabled=False)
                    p=r.get('spelling_context')
                    if p:
                        target.set_spelling_context(d.entity_id,keywords=p['keywords'],minimum_score=p.get('minimum_score',2.0),source=p['source'],
                                                   allow_internal_omission=p.get('allow_internal_omission',False),allow_near_spelling=p.get('allow_near_spelling',False))
                    elif 'spelling_context' in r:
                        target.clear_spelling_context(d.entity_id)
                return len(records)
            apply(staged)
            # One SQLite transaction applies only the staged entity tables.
            with store._connect() as connection:
                connection.execute('BEGIN IMMEDIATE')
                if contents(connection)!=original:
                    raise RuntimeError('Entity database changed during validation; import cancelled without applying changes')
                connection.execute('ATTACH DATABASE ? AS staged',(str(temporary),))
                staged_columns={r[1] for r in connection.execute('PRAGMA staged.table_info(entity_spelling_contexts)')}
                current_columns={r[1] for r in connection.execute('PRAGMA main.table_info(entity_spelling_contexts)')}
                for option in ('allow_internal_omission','allow_near_spelling'):
                    if option in staged_columns and option not in current_columns:
                        connection.execute('ALTER TABLE entity_spelling_contexts ADD COLUMN '+option+
                            ' INTEGER NOT NULL DEFAULT 0 CHECK ('+option+' IN (0,1))')
                # Old copies may append the two optional fields in different
                # orders. Name columns explicitly when copying context rows.
                for table in ['entity_spelling_contexts','entity_aliases','protected_entities']:
                    connection.execute('DELETE FROM '+table)
                for table in ['protected_entities','entity_aliases','entity_spelling_contexts']:
                    columns=[r[1] for r in connection.execute('PRAGMA staged.table_info('+table+')')]
                    names=','.join(columns)
                    connection.execute('INSERT INTO '+table+' ('+names+') SELECT '+names+' FROM staged.'+table)
            print(json.dumps(dict(imported=len(records)),ensure_ascii=False))
    else:
        records=[]
        for d in store.list_entities(include_disabled=True):
            p=d.spelling_context
            records.append(dict(entity_id=d.entity_id,canonical_text=d.canonical_text,entity_type=d.entity_type,
                domain=d.domain,normalization_policy=d.normalization_policy,aliases=list(d.aliases),priority=d.priority,
                enabled=d.enabled,source=d.source,spelling_context=None if p is None else dict(
                    keywords=dict(p.keywords),minimum_score=p.minimum_score,source=p.source,
                    allow_internal_omission=p.allow_internal_omission,allow_near_spelling=p.allow_near_spelling)))
        text=json.dumps(records,ensure_ascii=False,indent=2)+'\n'
        if args.action=='export':args.output.write_text(text)
        else:print(text,end='')


if __name__=='__main__':main()
