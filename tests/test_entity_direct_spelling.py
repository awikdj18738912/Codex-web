"""Direct glossary spelling requires an explicitly authorized entity boundary."""

import asyncio

from system.entity_audio_repair import EntityAudioRepair
from system.entity_store import EntityDefinition
from system.entity_direct_spelling import DirectEntitySpeller


def entity(canonical, entity_id=1, policy="normalize", enabled=True):
    return EntityDefinition(
        entity_id, canonical, "PERSON", "general", policy, 0, (), enabled=enabled
    )


def patches(text, definitions, *, final=False):
    return DirectEntitySpeller(definitions).select(text, final=final)


class FakeBackend:
    """Primary ASR only; it never claims to apply a context prompt."""

    def __init__(self, texts):
        self.texts = iter(texts)
        self.current = ""
        self.starts = []

    async def __call__(self, endpoint, session_id=None, data=b"", params=None):
        if endpoint == "/stream/start":
            self.starts.append(dict(params or {}))
            return {"session_id": "primary", "context_applied": False}
        if endpoint == "/stream/cancel":
            return {"cancelled": True}
        if endpoint == "/stream/chunk":
            self.current = next(self.texts)
        elif endpoint == "/stream/finish":
            remaining = list(self.texts)
            if remaining:
                self.current = remaining[-1]
                self.texts = iter(())
        return {
            "text": self.current,
            "completed_segments": [{"segment_id": 1, "text": self.current}],
        }


async def stream(backend, definitions, chunks=(b"audio",)):
    repair = EntityAudioRepair(backend, definitions)
    ready = await repair.request("/stream/start", params={"language": "Chinese"})
    outputs = []
    for chunk in chunks:
        outputs.append(await repair.request("/stream/chunk", ready["session_id"], chunk))
    outputs.append(await repair.request("/stream/finish", ready["session_id"]))
    return outputs


def test_declared_homophone_is_replaced_only_at_a_complete_entity_boundary():
    result = patches("记者采访张炜。", [entity("张伟")])
    assert len(result) == 1
    assert {key: result[0][key] for key in (
        "start", "end", "observed", "canonical", "entity_id", "evidence_source", "direct_rule"
    )} == {
        "start": 4, "end": 6, "observed": "张炜", "canonical": "张伟",
        "entity_id": 1, "evidence_source": "glossary_spelling", "direct_rule": "exact_sound",
    }


def test_protects_complete_registered_terms_that_contain_or_join_the_candidate():
    definitions = [entity("张伟"), entity("张维度量", 2, "preserve"), entity("扩张危机", 3, "preserve")]
    for text in ("这张位置图", "张维度量", "扩张危机"):
        assert patches(text, definitions) == []


def test_longer_registered_name_is_not_cut_into_shorter_name():
    definitions = [entity("张伟"), entity("张伟明", 2, "preserve")]
    assert patches("报道张炜明。", definitions) == []


def test_ambiguous_homophones_without_priority_are_rejected():
    assert patches("记者采访张炜。", [entity("张伟"), entity("张炜", 2)]) == []


def test_disabled_or_preserve_entries_protect_but_do_not_authorize_rewriting():
    for definition in (entity("张伟", policy="preserve"), entity("张伟", enabled=False)):
        assert patches("记者采访张炜。", [definition]) == []


def test_near_sound_accepts_a_single_letter_difference_in_one_syllable():
    cases = [
        ("毛中刚", "毛宗岗"),  # zhong/zong
        ("苏格拉你", "苏格拉底"),  # ni/di
    ]
    for observed, canonical in cases:
        result = patches("报道" + observed + "。", [entity(canonical)])
        assert len(result) == 1
        assert result[0]["observed"] == observed
        assert result[0]["canonical"] == canonical
        assert result[0]["direct_rule"] == "near_sound"


def test_near_sound_requires_unique_candidate_and_rejects_competing_spellings():
    definitions = [entity("毛宗刚"), entity("毛中康", 2)]
    assert patches("报道毛中刚。", definitions) == []


def test_non_normalize_longer_entity_still_guards_its_registered_span():
    definitions = [entity("张伟"), entity("张炜明", 2, policy="preserve")]
    assert patches("报道张炜明。", definitions) == []


def test_incomplete_trailing_candidate_waits_for_final_input():
    definition = entity("张伟")
    assert patches("记者采访张炜", [definition], final=False) == []
    result = patches("记者采访张炜", [definition], final=True)
    assert len(result) == 1
    assert result[0]["observed"] == "张炜"
    assert result[0]["direct_rule"] == "exact_sound"


def test_audio_repair_direct_patch_keeps_partial_and_final_segments_aligned_without_context_replay():
    async def scenario():
        text = "记者采访张炜。"
        backend = FakeBackend([text, text])
        partial, final = await stream(backend, [entity("张伟")])
        assert partial["text"] == final["text"] == "记者采访张伟。"
        assert partial["completed_segments"] == final["completed_segments"]
        assert partial["completed_segments"][0]["text"] == partial["text"]
        assert partial["entity_audio_repair_revision"] == final["entity_audio_repair_revision"] == 1
        assert len(backend.starts) == 1
        assert all("context" not in params for params in backend.starts)
    asyncio.run(scenario())


def test_trailing_candidate_waits_then_registered_long_name_is_not_rewritten():
    async def scenario():
        backend = FakeBackend(["我认识张炜", "我认识张炜明。", "我认识张炜明。"])
        definitions = [entity("张伟"), entity("张伟明", 2, "preserve")]
        partial, updated, final = await stream(backend, definitions, chunks=(b"one", b"two"))
        assert partial["text"] == "我认识张炜"
        assert updated["text"] == final["text"] == "我认识张炜明。"
        assert updated["entity_audio_repair_revision"] == final["entity_audio_repair_revision"] == 0
    asyncio.run(scenario())


def test_direct_patch_is_withdrawn_when_next_hypothesis_forms_a_longer_registered_name():
    async def scenario():
        backend = FakeBackend(["记者采访张炜。", "记者采访张炜明。", "记者采访张炜明。"])
        definitions = [entity("张伟"), entity("张炜明", 2, "preserve")]
        first, changed, final = await stream(backend, definitions, chunks=(b"one", b"two"))
        assert first["text"] == "记者采访张伟。"
        assert first["entity_audio_repair_revision"] == 1
        assert changed["text"] == final["text"] == "记者采访张炜明。"
        assert changed["entity_audio_repair_revision"] == final["entity_audio_repair_revision"] == 2
        assert changed["completed_segments"][0]["text"] == changed["text"]
    asyncio.run(scenario())


def test_near_sound_glossary_patch_does_not_start_an_audio_review():
    async def scenario():
        text = "报道毛中刚。"
        backend = FakeBackend([text, text])
        partial, final = await stream(backend, [entity("毛宗岗")])
        assert partial["text"] == final["text"] == "报道毛宗岗。"
        assert final["entity_audio_repair"]["changes"][0]["direct_rule"] == "near_sound"
        assert len(backend.starts) == 1
        assert all("context" not in params for params in backend.starts)
    asyncio.run(scenario())


def test_ordinary_word_about_is_not_rewritten_as_goddess_guanyu():
    async def scenario():
        text = "关于这件事我们再讨论。"
        backend = FakeBackend([text, text])
        partial, final = await stream(backend, [entity("关羽")])
        assert partial["text"] == final["text"] == text
        assert final["entity_audio_repair_revision"] == 0
        assert not final["entity_audio_repair"]["changes"]
    asyncio.run(scenario())



def test_unknown_long_name_is_not_split_without_registering_it():
    assert patches("记者采访张炜明。", [entity("张伟")]) == []


def test_toneless_coincidence_is_not_a_homophone():
    assert patches("在经济下行的背景下。", [entity("北京")]) == []
