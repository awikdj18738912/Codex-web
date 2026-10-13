from system.repetition_preservation import repetition_preservation_violations as violations
from system.refinement_guard import (reject_reasons,find_repetition_review_candidates,
    apply_repetition_review_decisions,preserve_safe_repetition_edits)
from system.deterministic_cleanup import (clean_transcript_deterministically,
    collapse_repeated_short_utterances,collapse_repeated_comma_items)
from system.repetition_preservation import restore_preserved_repetitions


def test_distant_named_reference_restores_only_lost_copy():
    raw='接收完成，接收完成。接下来登记编号，记录存入档案盒。前面的接收完成说了两遍，需要原样保留。'
    target='接收完成。接下来登记编号。记录存入档案盒。前面的接收完成说了两遍，需要原样保留。'
    result, edits=restore_preserved_repetitions(raw,target)
    assert result=='接收完成，接收完成。接下来登记编号。记录存入档案盒。前面的接收完成说了两遍，需要原样保留。'
    assert len(edits)==1
    assert restore_preserved_repetitions(raw,result)==(result,[])


def test_recovery_retains_entity_numeric_and_unrelated_repairs():
    raw='接收完成，接收完成。联系联系郭新，金额三百元。前面的接收完成说了两遍。'
    target='接收完成。联系郭鑫，金额300元。前面的接收完成说了两遍。'
    result, edits=restore_preserved_repetitions(raw,target)
    assert result=='接收完成，接收完成。联系郭鑫，金额300元。前面的接收完成说了两遍。'
    assert edits
    # Normalized content within a repeated span must never be rolled back.
    assert restore_preserved_repetitions('三百元，三百元。原样保留。','300元。原样保留。')==('300元。原样保留。',[])


def test_recovery_does_not_attach_generic_distant_cues_or_restore_replacement():
    raw='接收完成，接收完成。登记编号。存入档案盒。要原样保留。'
    assert restore_preserved_repetitions(raw,raw.replace('接收完成，','',1))[1]==[]
    assert restore_preserved_repetitions('很急，很急。','非常急。')==('非常急。',[])
    assert restore_preserved_repetitions('很急，很急，很急。','很急，很急。')[0]=='很急，很急，很急。'


def test_named_reference_selects_nearest_same_unit_instance():
    raw='完成，完成。登记编号。完成，完成。存入档案盒。前面的完成说了两遍。'
    target='完成。登记编号。完成。存入档案盒。前面的完成说了两遍。'
    assert restore_preserved_repetitions(raw,target)[0]=='完成。登记编号。完成，完成。存入档案盒。前面的完成说了两遍。'


def test_named_reference_tolerates_asr_punctuation_without_protecting_other_stutter():
    from system.repetition_preservation import protected_repetitions
    for punctuation in ('，', ',', '：', ' ', '”'):
        raw='已经备案，已经备案。接着确认，确认货位。资料交给主管。前面的已经备案'+punctuation+'说了两遍，需要原样保留。'
        assert [p.unit for p in protected_repetitions(raw)]==['已经备案']
        target='已经备案。接着确认货位。资料交给主管。前面的已经备案'+punctuation+'说了两遍，需要原样保留。'
        assert restore_preserved_repetitions(raw,target)[0]==target.replace('已经备案。','已经备案，已经备案。',1)


def test_named_reference_to_missing_pair_never_protects_unrelated_pair():
    from system.repetition_preservation import protected_repetitions
    assert protected_repetitions('接着确认，确认货位。前面的凭证收到，说了两遍，需要原样保留。')==()


def test_source_clause_noun_reuse_survives_local_removal_and_boundary_loss():
    for unit,verb in [('索引','交给值班员'),('名单','送到调度室'),('附件','需要归档'),('样品','已经封存')]:
        raw=f'随后检查{unit}，{unit}{verb}。'
        merged=f'随后检查{unit}{verb}。'
        assert 'cross_clause_noun_reuse' in violations(raw,merged)
        assert restore_preserved_repetitions(raw,merged)[0]==raw
        # Even when both lexical copies survived, restore the missing boundary.
        no_separator=f'随后检查{unit}{unit}{verb}。'
        result,repairs=restore_preserved_repetitions(raw,no_separator)
        assert result==raw and any(r['reason']=='source_clause_boundary_restoration' for r in repairs)


def test_clause_protection_is_not_blanket_punctuation_or_noun_protection():
    for raw,edited in [('接着确认，确认货位。','接着确认货位。'),
                       ('请先检查，检查设备。','请先检查设备。'),
                       ('随后检查名单名单送到办公室。','随后检查名单送到办公室。'),
                       ('请核对名单，名单上的编号。','请核对名单上的编号。')]:
        assert not violations(raw,edited)
        assert restore_preserved_repetitions(raw,edited)==(edited,[])


def test_source_edit_guard_tracks_individual_spans_and_retains_unrelated_repairs():
    from system.repetition_preservation import source_repetition_edit_violations
    raw='核对名单，名单交给主管。检查附件，附件送到仓库。请确认确认金额。'
    before='核对名单交给主管。检查附件附件送到仓库。请确认确认金额。'
    assert source_repetition_edit_violations(raw,before,before.replace('确认确认','确认'))==()
    assert source_repetition_edit_violations(raw,before,before.replace('附件附件','附件'))==('cross_clause_noun_reuse',)


def test_clause_restoration_preserves_entity_numeric_edits_and_existing_punctuation():
    raw='核对名单，名单交给郭心，金额三百元。'
    target='核对名单交给郭新，金额300元。'
    assert restore_preserved_repetitions(raw,target)[0]=='核对名单，名单交给郭新，金额300元。'
    assert restore_preserved_repetitions('核对名单。名单交给主管。','核对名单，名单交给主管。')[1]==[]
    assert restore_preserved_repetitions('联系郭心，郭心交给主管。','联系郭新交给主管。')==('联系郭新交给主管。',[])


def test_shared_guard_protects_multiple_semantic_shapes():
    examples=[('备件要一箱一箱核对。','备件要一箱核对。'),
              ('手续很麻烦很麻烦。','手续很麻烦。'),
              ('不要外传，不要外传。','不要外传。'),
              ('已经核验，已经核验。原样记录这句话。','已经核验。原样记录这句话。')]
    for raw,edited in examples:
        assert violations(raw,edited)
        assert 'repetition_intent_loss' in reject_reasons(raw,edited)


def test_exact_repetition_local_override_cannot_bypass_preservation():
    raw='这个步骤非常非常紧急，马上处理。'
    candidates=find_repetition_review_candidates(raw)
    edited,records=apply_repetition_review_decisions(raw,candidates,[dict(index=c.index,action='remove') for c in candidates])
    assert edited==raw and not any(r['applied'] for r in records)


def test_salvage_keeps_good_cleanup_next_to_prohibited_deletion():
    raw='这个步骤特别特别关键，请确认确认地址。'
    edited='这个步骤特别关键，请确认地址。'
    assert preserve_safe_repetition_edits(raw,edited)=='这个步骤特别特别关键，请确认地址。'


def test_cleanup_cannot_remove_emphasis_or_repeated_negation():
    for text in ['没收到，没收到。','特别关键！特别关键！','完成，完成。原样记录。']:
        assert clean_transcript_deterministically(text)==text
        assert collapse_repeated_short_utterances(text)==text
        assert collapse_repeated_comma_items(text)==text


def test_ordinary_repair_still_allowed():
    for raw,edited in [('联系联系供应商。','联系供应商。'),('工作工作群。','工作群。')]:
        assert not violations(raw,edited)
        assert not reject_reasons(raw,edited)
    assert clean_transcript_deterministically('呃，没收到，没收到。')=='没收到，没收到。'


def test_unrelated_emphasis_word_does_not_protect_stutter():
    assert not violations('请核对工作工作群，之后强调保留签名。','请核对工作群，之后强调保留签名。')


def test_explicit_intent_attaches_to_nearest_pair_only():
    raw='我我说手续困难困难，发言人特意说了两遍。'
    assert not violations(raw,'我说手续困难困难，发言人特意说了两遍。')
    assert violations(raw,'我我说手续困难，发言人特意说了两遍。')


def test_multiplicity_and_multiple_occurrences_must_survive():
    assert violations('很急，很急，很急。','很急，很急。')
    assert violations('很急很急。后面很急很急。','很急。后面很急很急。')
    assert not violations('很急，很急。','很急很急！')


def test_numeric_rule_recursion_cannot_hide_preserved_content():
    raw='不能交付2件，不能交付2件。'
    assert 'repetition_intent_loss' in reject_reasons(raw,'不能交付2件。',numeric_rule_mode=True)


def test_unprotected_later_copy_cannot_substitute_for_protected_occurrence():
    raw='完成，完成。原样记录这句话。另一个是完成，完成。'
    assert violations(raw,'完成。原样记录这句话。另一个是完成，完成。')
    assert not violations(raw,'完成，完成。原样记录这句话。另一个是完成。')


def test_explicit_count_is_not_limited_to_two_or_three():
    raw='完成完成。负责人强调了4次。'
    assert violations(raw,'完成。负责人强调了4次。')


def test_new_pair_collector_records_preflight_keep_with_legacy_fallback():
    from types import SimpleNamespace
    from experiments.scripts.collect_gate_pairs import local_review_native_action
    text='特别特别关键。'
    candidate=find_repetition_review_candidates(text)[0]
    assert local_review_native_action(SimpleNamespace(),text,candidate)=='refine'
    assert local_review_native_action(SimpleNamespace(repetition_preservation_violations=violations),text,candidate)=='keep'
