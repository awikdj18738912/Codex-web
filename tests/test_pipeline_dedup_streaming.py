import random
import string

from pipeline.src.pipeline.dedup import (
    StreamingDeduplicator,
    char_ngrams,
    deduplicate,
    jaccard_similarity,
)


def naive(items, threshold, text_key="input"):
    retained = []
    sets = []
    fragments = []
    for item in items:
        text = item.get(text_key, "")
        if not text:
            retained.append(item)
            continue
        grams = char_ngrams(text)
        fragment = bool((item.get("meta") or {}).get("is_fragment"))
        candidates = (
            idx for idx, old in enumerate(sets)
            if fragments[idx] == fragment and grams & old
        )
        if any(jaccard_similarity(grams, sets[idx]) >= threshold for idx in candidates):
            continue
        retained.append(item)
        sets.append(grams)
        fragments.append(fragment)
    return retained


def test_streaming_matches_naive_for_random_and_adversarial_inputs():
    rng = random.Random(31824)
    alphabet = string.ascii_lowercase[:8]
    texts = ["".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 50))) for _ in range(300)]
    texts += ["common-prefix-" + str(i) for i in range(40)]
    texts += ["identical-prefix-" + "x" * i for i in range(1, 15)]
    texts += ["same exact input", "same exact input", "", "x"]
    items = [
        {"input": text, "meta": {"is_fragment": i % 5 == 0}}
        for i, text in enumerate(texts)
    ]
    for threshold in (0.75, 1.0, 0.5, 0.0):
        expected = naive(items, threshold)
        assert deduplicate(items, threshold=threshold) == expected


def test_streaming_api_incremental_and_fragment_isolation():
    deduper = StreamingDeduplicator(0.75)
    full = {"input": "abcdefghijk"}
    fragment = {"input": "abcdefghijk", "meta": {"is_fragment": True}}
    assert deduper.add(full)
    assert not deduper.add({"input": "abcdefghijk"})
    assert deduper.add(fragment)
    assert not deduper.add({"input": "abcdefghijk", "meta": {"is_fragment": True}})
    assert deduper.add({"input": "", "meta": {}})


def test_deduplicate_respects_custom_text_key():
    items = [{"text": "abcdef"}, {"text": "abcdef"}]
    assert deduplicate(items, text_key="text") == items[:1]
