"""Opt-in, context-backed one-letter completion for short uppercase codes."""

from dataclasses import replace
import asyncio

from system.entity_direct_spelling import DirectEntitySpeller
from system.entity_store import EntityDefinition, EntitySpellingContext
from system.entity_audio_repair import EntityAudioRepair


def entity(canonical, entity_id=1, *, context=None, aliases=(), policy="normalize"):
    return EntityDefinition(
        entity_id, canonical, "TERM", "medical", policy, 0, tuple(aliases),
        spelling_context=context,
    )


def context(canonical, *, keywords, minimum=2, allow=True):
    return EntitySpellingContext(
        tuple(keywords), minimum, "verified glossary context", canonical,
        "TERM", "medical", allow_near_spelling=allow,
    )


def ilc_entities(*, policy=True, extra=()):
    definitions = []
    for i in range(1, 4):
        canonical = f"ILC{i}"
        ctx = context(canonical, keywords=(("淋巴", 1), ("细胞", 1))) if policy else None
        definitions.append(entity(canonical, i, context=ctx))
    return [*definitions, *extra]


def patches(text, definitions, *, final=True):
    return DirectEntitySpeller(definitions).select(text, final=final)


def test_contextual_single_letter_deletions_restore_ilc_numbered_codes():
    for observed, canonical in (("LC一", "ILC1"), ("LC二", "ILC2"), ("LC三", "ILC3")):
        result = patches(f"淋巴细胞检测{observed}。", ilc_entities())
        assert len(result) == 1
        assert result[0]["observed"] == observed
        assert result[0]["canonical"] == canonical
        assert result[0]["direct_rule"] == "context_code_indel"


def test_fast_single_letter_deletion_needs_independent_context():
    definition = entity("FAS", context=context("FAS", keywords=(("APO-1", 2), ("蛋白", 1), ("凋亡", 1))))
    assert patches("APO-1相关检测FAST阳性。", [definition])[0]["canonical"] == "FAS"
    assert patches("蛋白和凋亡检测FAST阳性。", [definition])[0]["canonical"] == "FAS"


def test_context_route_does_not_create_typo_aliases_or_ignore_opt_in():
    definitions = ilc_entities()
    speller = DirectEntitySpeller(definitions)
    assert all("LC" not in definition.aliases for definition in definitions)
    assert patches("淋巴细胞LC一。", ilc_entities(policy=False)) == []
    denied = ilc_entities()
    denied[0] = replace(denied[0], spelling_context=replace(
        denied[0].spelling_context, allow_near_spelling=False))
    assert patches("淋巴细胞LC一。", denied) == []
    assert speller.select("没有相关上下文LC一。") == []


def test_code_casing_word_boundaries_and_unsupported_numbers_are_preserved():
    definitions = ilc_entities()
    for text in (
        "lymphocyte LC一。", "淋巴细胞fast。", "淋巴细胞FASTING。",
        "淋巴细胞BREAKFAST。", "淋巴细胞XLC一。", "淋巴细胞_LC一。",
        "淋巴细胞LC十。", "淋巴细胞LC四。",
    ):
        assert patches(text, definitions) == [], text
    fas = entity('FAS', context=context('FAS', keywords=(("APO-1", 2),)))
    for text in ('APO-1相关fast。', 'APO-1相关FASTING。',
                 'APO-1相关BREAKFAST。', 'APO-1相关_FAST。', 'FAST检查。'):
        assert patches(text, [fas]) == [], text


def test_two_missing_letters_and_substitutions_are_not_completed():
    definitions = ilc_entities(extra=[entity("FAST", 4, context=context(
        "FAST", keywords=(("APO-1", 2),)))])
    for text in ("淋巴细胞C一。", "淋巴细胞LA一。", "APO-1相关XAST。", "APO-1相关FIST。"):
        assert patches(text, definitions) == [], text


def test_registered_literal_and_competing_canonical_codes_block_completion():
    definitions = ilc_entities(extra=[entity("LC1", 10, policy="preserve")])
    assert patches("淋巴细胞LC1。", definitions) == []
    assert patches("淋巴细胞LC一。", definitions) == []
    assert patches("淋巴细胞LC一。", [entity("ILC1", 1, context=context(
        "ILC1", keywords=(("淋巴", 1), ("细胞", 1)))),
        entity("ALC1", 2, context=context("ALC1", keywords=(("淋巴", 1), ("细胞", 1))))]) == []


def test_longer_registered_phrase_and_nonfinal_tail_are_protected():
    definitions = ilc_entities(extra=[entity("LC1相关蛋白", 9, policy="preserve")])
    assert patches("淋巴细胞LC一相关蛋白。", definitions) == []
    assert patches("淋巴细胞LC一", definitions, final=False) == []


def test_streaming_context_repairs_keep_segment_offsets_and_withdraw_revisions():
    class Backend:
        text = '淋巴细胞包括LC一、LC二、LC三。蛋白FAST又叫APO-1。'

        async def __call__(self, endpoint, session_id=None, data=b'', params=None):
            if endpoint == '/stream/start':
                return {'session_id': 'primary', 'context_applied': False}
            return {'text': self.text, 'completed_segments': [
                {'segment_id': 1, 'text': self.text}]}

    async def scenario():
        backend = Backend()
        definitions = ilc_entities(extra=[entity('FAS', 4, context=context(
            'FAS', keywords=(("APO-1", 2),)))])
        repair = EntityAudioRepair(backend, definitions, short_finish_max_seconds=0)
        try:
            await repair.request('/stream/start')
            partial = await repair.request('/stream/chunk', 'primary', b'audio')
            expected = '淋巴细胞包括ILC1、ILC2、ILC3。蛋白FAS又叫APO-1。'
            assert partial['text'] == expected
            assert partial['completed_segments'][0]['text'] == expected
            # A later ASR revision reveals a longer English word. The old
            # source-anchored correction must disappear before final output.
            backend.text = backend.text.replace('FAST', 'FASTING')
            final = await repair.request('/stream/finish', 'primary')
            assert final['text'] == expected.replace('FAS又叫', 'FASTING又叫')
            assert final['completed_segments'][0]['text'] == final['text']
        finally:
            repair.close()
    asyncio.run(scenario())
