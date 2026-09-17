"""Common Mandarin quantifiers used by transcript-safety rules.

Classifier inventories are productive rather than closed.  This module is
therefore deliberately used as a *safety vocabulary* (protecting likely
reduplicated classifiers), not as a requirement that every classifier be
listed before it can occur in a transcript.
"""

from __future__ import annotations


# The list covers the common noun, collective, container, shape, time,
# measure, and event classifiers found in modern Mandarin reference lists.
# Multi-character entries document the inventory; the reduplication guard uses
# only one-character entries because the protected form is ``一XX``/``一X X``.
COMMON_QUANTIFIERS = frozenset(
    """
    把 瓣 本 部 册 出 处 床 道 点 顶 锭 栋 朵 分 封 幅 副 杆 个 根 管 户 级
    剂 家 架 间 件 节 具 句 棵 颗 口 块 粒 辆 列 轮 枚 门 面 名 盘 匹 篇
    片 期 曲 扇 身 首 艘 所 台 堂 条 贴 听 挺 头 尾 位 项 眼 页 员 则 爿
    盏 张 枝 支 只 帧 株 桩 幢 宗 尊 座 班 帮 笔 队 对 股 伙 排 批 群 双
    套 窝 系列 组 行列 家族 人马 阵营 份
    尺 寸 度 吨 分 伏 公斤 公里 公顷 毫米 毫升 斤 克 里 两 立方米 米 亩
    匹 平方公里 平方米 千瓦 顷 升 微米 元
    包 杯 车 池 袋 缸 罐 盒 壶 窖 筐 篓 篮 盘 盆 瓶 勺 盅 钵 坛 桶 碗 箱 桌
    层 串 丛 撮 滴 叠 堵 段 堆 垛 股 挂 行 截 节 卷 捆 绺 摞 缕 排 派
    泡 匹 片 撇 腔 束 丝 摊 滩 团 线 泓 汪 弯 抹 簇 波 茬 畦 垄 蓬 沓 绳 痕
    脉 弧 袭 笼 彪
    次 遍 场 顿 趟 回 下 番 阵 通 轮 转 番
    年 月 日 天 夜 秒 分钟 分钟 小时 刻 会儿 顿 人
    级 类 门 样 种 番 式 色 款
    点 点儿 些 许
    成 倍 分
    人次 架次 辆次 台次 台班 吨公里 秒立方米
    """.split()
)

# Numeral/determiner prefixes that license a distributive classifier form:
# ``一朵朵``, ``两排排``, ``各家家`` and so on.  ``数`` is included for
# forms such as ``数道道`` encountered in documentary-style speech.
QUANTIFIER_REDUP_PREFIXES = frozenset("一两几各每众逐数")

REDUPLICABLE_QUANTIFIER_CHARS = frozenset(
    token for token in COMMON_QUANTIFIERS if len(token) == 1
)


def is_prefixed_reduplicated_quantifier(text: str, pair_start: int) -> bool:
    """Return whether ``text[pair_start:pair_start+2]`` is a classifier AA.

    The prefix is intentionally structural, so newly encountered classifiers
    can be protected by extending the vocabulary without changing every
    cleanup rule.  Out-of-range and non-Chinese input simply return ``False``.
    """

    if not has_reduplicated_classifier_shape(text, pair_start):
        return False
    return text[pair_start] in REDUPLICABLE_QUANTIFIER_CHARS


def has_reduplicated_classifier_shape(text: str, pair_start: int) -> bool:
    """Return whether a determiner prefixes any Chinese ``AA`` classifier form.

    The structural fallback intentionally accepts an unseen one-character
    classifier.  Mandarin classifier inventories are open and regional, so a
    finite vocabulary must not make an unseen form vulnerable to AA deletion.
    """

    if pair_start < 1 or pair_start + 1 >= len(text):
        return False
    character = text[pair_start]
    return (
        text[pair_start + 1] == character
        and character >= "\u3400"
        and character <= "\u9fff"
        and text[pair_start - 1] in QUANTIFIER_REDUP_PREFIXES
    )


def is_quantifier_reduplication_deletion(
    text: str, start: int, end: int, character: str
) -> bool:
    """Return whether deleting one ``character`` removes a protected ``X X``."""

    if len(character) != 1:
        return False
    if text[start - 1 : start] == character:
        pair_start = start - 1
    elif text[end : end + 1] == character:
        pair_start = start
    else:
        return False
    return has_reduplicated_classifier_shape(text, pair_start)


__all__ = [
    "COMMON_QUANTIFIERS",
    "QUANTIFIER_REDUP_PREFIXES",
    "REDUPLICABLE_QUANTIFIER_CHARS",
    "has_reduplicated_classifier_shape",
    "is_prefixed_reduplicated_quantifier",
    "is_quantifier_reduplication_deletion",
]
