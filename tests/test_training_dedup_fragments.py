from pipeline.src.pipeline.dedup import deduplicate


def test_similar_full_and_partial_keep_their_distinct_targets():
    full = {"input": "请把本周所有销售订单按城市和日期导出成表格，并发送给财务组，然后确认。", "meta": {"is_fragment": False}}
    partial = {"input": "请把本周所有销售订单按城市和日期导出成表格，并发送给财务组，", "meta": {"is_fragment": True}}
    assert deduplicate([full, partial, dict(full), dict(partial)]) == [full, partial]
