"""Explicit glossary spellings across Chinese, Latin, and numeric forms."""
import asyncio

from system.entity_direct_spelling import DirectEntitySpeller
from system.entity_audio_repair import EntityAudioRepair
from system.entity_store import EntityDefinition


def entity(canonical, entity_id=1, *, aliases=(), policy="normalize", enabled=True):
    return EntityDefinition(entity_id, canonical, "TERM", "medical", policy, 0,
                            tuple(aliases), enabled=enabled)


def patches(text, definitions, *, final=False):
    return DirectEntitySpeller(definitions).select(text, final=final)


def test_short_chinese_number_spellings_and_nfkc_latin_forms_normalize():
    definitions = [entity("B1细胞", 1), entity("B2细胞", 2)]
    cases = [
        ("B一细胞", "B1细胞"),
        ("B 二细胞", "B2细胞"),
        ("Ｂ１细胞", "B1细胞"),
        ("b1细胞", "B1细胞"),
    ]
    for source, canonical in cases:
        result = patches("检验提示" + source + "异常。", definitions)
        assert len(result) == 1
        assert result[0]["observed"] == source
        assert result[0]["canonical"] == canonical


def test_explicit_latin_alias_matches_spaced_acronym_and_normalizes_tcr_bcr():
    definitions = [entity("T细胞受体", 1, aliases=("TCR受体",)),
                   entity("B细胞受体", 2, aliases=("BCR受体",))]
    for source, canonical in (("T C R受体", "T细胞受体"),
                              ("B C R受体", "B细胞受体")):
        result = patches("检测到" + source + "阳性。", definitions)
        assert len(result) == 1
        assert result[0]["observed"] == source
        assert result[0]["canonical"] == canonical


def test_explicit_neutrophil_alias_and_near_rule_are_distinct_authorities():
    canonical = entity("嗜中性粒细胞", aliases=("中性粒细胞",))
    result = patches("样本中性粒细胞增多。", [canonical])
    assert len(result) == 1 and result[0]["canonical"] == "嗜中性粒细胞"
    near = patches("样本中心粒细胞增多。", [canonical])
    assert len(near) == 1 and near[0]["canonical"] == "嗜中性粒细胞"
    assert near[0]["direct_rule"] == "near_sound"
    assert patches("样本中性粒细胞增多。", [entity("嗜中性粒细胞", aliases=())]) == []


def test_latin_word_boundaries_and_registered_longer_terms_are_protected():
    assert patches("检出ILC10细胞。", [entity("ILC1细胞")]) == []
    assert patches("检出XILC1细胞。", [entity("ILC1细胞")]) == []
    # A longer registered preserve term protects its full span from a short
    # canonical candidate; the independent-token case must also stay intact.
    definitions = [entity("B1细胞"), entity("B1细胞表面受体", 2, policy="preserve")]
    assert patches("检测B一细胞表面受体。", definitions) == []


def test_clause_breaks_ambiguity_and_unsupported_numeric_guesses_are_rejected():
    assert patches("检出T,C,R受体。", [entity("T细胞受体", aliases=("TCR受体",))]) == []
    # The same normalized evidence key cannot choose between canonical owners.
    assert patches("检出T C R受体。", [
        entity("T细胞受体", aliases=("TCR受体",)),
        entity("T淋巴细胞受体", 2, aliases=("TCR受体",)),
    ]) == []
    assert patches("B7细胞异常。", [entity("B1细胞")]) == []
    assert patches("naive B细胞异常。", [entity("B细胞")]) == []
    assert patches("FAST检查完成。", [entity("FAS")]) == []
    assert patches("these two words are separate。", [entity("twowords")]) == []
    assert patches("检出ABC DEF。", [entity("ABCDEF")]) == []


def test_nonfinal_trailing_candidate_waits_for_right_context_or_finish():
    definition = entity("T细胞受体", aliases=("TCR受体",))
    assert patches("发现T C R受体", [definition], final=False) == []
    assert len(patches("发现T C R受体", [definition], final=True)) == 1
    assert len(patches("发现T C R受体异常", [definition], final=False)) == 1


class FakeBackend:
    def __init__(self, text):
        self.text = text
        self.starts = []

    async def __call__(self, endpoint, session_id=None, data=b"", params=None):
        if endpoint == "/stream/start":
            self.starts.append(dict(params or {}))
            return {"session_id": "primary", "context_applied": False}
        if endpoint == "/stream/cancel":
            return {"cancelled": True}
        return {"text": self.text,
                "completed_segments": [{"segment_id": 1, "text": self.text}]}


def test_english_only_definition_keeps_audio_repair_enabled():
    async def scenario():
        repair = EntityAudioRepair(FakeBackend(""), [entity("FAS")])
        ready = await repair.request("/stream/start", params={"language": "Chinese"})
        assert repair.enabled is True
        assert ready["session_id"] == "primary"
        await repair.request("/stream/cancel", "primary")
    asyncio.run(scenario())


def test_direct_latin_alias_replacement_keeps_segments_aligned_without_context_replay():
    async def scenario():
        text = "检测到T C R受体阳性。"
        backend = FakeBackend(text)
        repair = EntityAudioRepair(
            backend, [entity("T细胞受体", aliases=("TCR受体",))]
        )
        ready = await repair.request("/stream/start", params={"language": "Chinese"})
        partial = await repair.request("/stream/chunk", ready["session_id"], b"audio")
        final = await repair.request("/stream/finish", ready["session_id"])
        expected = "检测到T细胞受体阳性。"
        assert partial["text"] == final["text"] == expected
        assert partial["completed_segments"] == final["completed_segments"]
        assert final["completed_segments"][0]["text"] == expected
        assert len(backend.starts) == 1
        assert all("context" not in params for params in backend.starts)
    asyncio.run(scenario())
