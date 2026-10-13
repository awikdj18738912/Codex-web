from experiments.scripts.export_operation_effect_labels import operation_effect


def test_known_cleanup_not_hidden_by_unrelated_name_difference():
    effect,basis,target=operation_effect('uncertain','需要需要周清确认','需要周清确认',
                                       '需要需要周青确认','需要周青确认')
    assert effect=='benefit' and target=='需要周青确认'
    assert basis.endswith('unverified_audio')


def test_extra_unsupported_change_stays_uncertain():
    assert operation_effect('uncertain','需要需要周清确认','需要周清确认',
                            '需要需要周青确认','需要王青确认')[0]=='uncertain'


def test_substituted_or_unaligned_intended_edits_stay_uncertain():
    assert operation_effect('uncertain','五元','六元','五元','6元')[0]=='uncertain'
    assert operation_effect('uncertain','需要需要确认','需要确认','确认','需要确认')[0]=='uncertain'


def test_existing_full_text_harm_is_never_promoted():
    assert operation_effect('harm','特别特别重要','特别特别重要','特别特别重要','特别重要')[0]=='harm'
