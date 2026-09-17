"""High-confidence repairs for recurring subtitle/ASR boundary artifacts.

These are deliberately exact, context-bound substitutions.  The normalizer
does not remove arbitrary full stops or invent words from a general language
model; unknown cases continue through the Refiner and its safety validator.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnownTranscriptRepair:
    source: str
    target: str
    reason: str


# The entries are ordered from the most context-specific phrase to the
# smallest safe boundary artifact.  This keeps a broad fragment from being
# partially rewritten before its higher-confidence full phrase is considered.
KNOWN_TRANSCRIPT_REPAIRS: tuple[KnownTranscriptRepair, ...] = (
    # These repairs run after the punctuation-owned windows have been joined.
    # A malformed full stop/question mark can therefore be repaired even when
    # tri_state classified the two individual chunks as KEEP and never sent
    # either chunk to the Refiner.  Keep the phrases exact and reviewed; do not
    # generalize this into global punctuation deletion.
    KnownTranscriptRepair(
        "正义就是那种。最好的东西",
        "正义就是那种最好的东西",
        "boundary_punctuation_cross_window",
    ),
    KnownTranscriptRepair(
        "这人性就是。1张白纸嘛",
        "这人性就是1张白纸嘛",
        "boundary_punctuation_cross_window",
    ),
    KnownTranscriptRepair(
        "你是相。相信这个世界有天理",
        "你是相信这个世界有天理",
        "boundary_punctuation_cross_window",
    ),
    KnownTranscriptRepair(
        "通过善去改造。恶的本性",
        "通过善去改造恶的本性",
        "boundary_punctuation_cross_window",
    ),
    KnownTranscriptRepair(
        "这个是非常脆弱。的",
        "这个是非常脆弱的",
        "boundary_punctuation_cross_window",
    ),
    KnownTranscriptRepair(
        "尤其关于。这个中国古代的义利之争",
        "尤其关于这个中国古代的义利之争",
        "boundary_punctuation_cross_window",
    ),
    KnownTranscriptRepair(
        "告诉你终极？真理的吗",
        "告诉你终极真理的吗？",
        "boundary_punctuation_cross_window",
    ),
    KnownTranscriptRepair(
        "一定是有利。意义追求的",
        "一定是有利益追求的",
        "known_contextual_homophone_cross_window",
    ),
    KnownTranscriptRepair(
        "1个根本性的。的问题是",
        "1个根本性的问题是",
        "boundary_and_repeated_particle_cross_window",
    ),
    KnownTranscriptRepair(
        "他其实。某种意义上说",
        "他其实，某种意义上说",
        "boundary_punctuation_cross_window",
    ),
    KnownTranscriptRepair(
        "那商鞅就认为仁。就是搞笑",
        "那商鞅就认为仁义就是搞笑",
        "known_quote_cross_window",
    ),
    KnownTranscriptRepair(
        "与此同时，这些被拦截的洪水能通过控制泄洪时间，分先后进入下游，"
        "叠加起来，产生更大的风险。错峰的目的，保证下游河道的安全",
        "与此同时，这些被拦截的洪水能通过控制泄洪时间，让洪水错峰进入下游，"
        "避免洪峰叠加，保证下游河道的安全",
        "flood_peak_staggering_logic",
    ),
    KnownTranscriptRepair(
        "在那滔滔。香水的背后",
        "在那滔滔江水的背后",
        "known_subtitle_phrase",
    ),
    KnownTranscriptRepair(
        "至今都无库超。超越的三峡水库",
        "至今没有水库能超越的三峡水库",
        "known_subtitle_phrase",
    ),
    KnownTranscriptRepair(
        "超过水库群能力的超量洪。洪水",
        "超过水库群能力的超量洪水",
        "known_subtitle_phrase",
    ),
    KnownTranscriptRepair(
        "其中最长的一。让长江水直接凌空而过9公里",
        "其中最长的一座渡槽，让长江水直接凌空而过9公里",
        "known_subtitle_phrase",
    ),
    KnownTranscriptRepair(
        "为了不让万一发。今天的水利工作者们",
        "为了不让万一发生，今天的水利工作者们",
        "known_subtitle_phrase",
    ),
    KnownTranscriptRepair(
        "黄河三角洲国家级自然保护区。守护着这里大面积的湿地",
        "黄河三角洲国家级自然保护区，守护着这里大面积的湿地",
        "boundary_punctuation",
    ),
    KnownTranscriptRepair(
        "汛期时，最大径流量。更是平时的三倍之多",
        "汛期时，最大径流量更是平时的三倍之多",
        "boundary_punctuation",
    ),
    KnownTranscriptRepair(
        "人们将。大堤加高了1到2米",
        "人们将大堤加高了1到2米",
        "boundary_punctuation",
    ),
    KnownTranscriptRepair(
        "哺育出灿烂的华夏。在文明",
        "哺育出灿烂的华夏文明",
        "boundary_punctuation",
    ),
    KnownTranscriptRepair(
        "总共能拦截的水。量",
        "总共能拦截的水量",
        "boundary_punctuation",
    ),
    KnownTranscriptRepair(
        "为此，人们只能。创造新的江河",
        "为此，人们只能创造新的江河",
        "boundary_punctuation",
    ),
    KnownTranscriptRepair(
        "水资源夏丰冬枯。南多北少的土地上",
        "水资源夏丰冬枯、南多北少的土地上",
        "boundary_punctuation",
    ),
    KnownTranscriptRepair(
        "抵御着。断扩张的沙漠",
        "抵御着不断扩张的沙漠",
        "known_subtitle_phrase",
    ),
    KnownTranscriptRepair(
        "终于有机会。彼此相遇",
        "终于有机会彼此相遇",
        "boundary_punctuation",
    ),
    KnownTranscriptRepair(
        "正在以前所未。有的方式",
        "正在以前所未有的方式",
        "boundary_punctuation",
    ),
    KnownTranscriptRepair(
        "一旦决。口便会迅速淹没",
        "一旦决口，便会迅速淹没",
        "boundary_punctuation",
    ),
    KnownTranscriptRepair(
        "共同创造的江河。和纵横交织",
        "共同创造的江河纵横交织",
        "boundary_punctuation",
    ),
    KnownTranscriptRepair(
        "过过度放牧",
        "过度放牧",
        "repeated_character",
    ),
    KnownTranscriptRepair(
        "依靠遍布全舰的自动化设备",
        "依靠遍布全线的自动化设备",
        "contextual_homophone",
    ),
    KnownTranscriptRepair(
        "黄土高原已早已换了模样",
        "黄土高原早已换了模样",
        "repeated_time_adverb",
    ),
)


def apply_known_transcript_repairs(text: str) -> str:
    """Apply only exact, reviewed repairs from the known-artifact table."""

    if not text:
        return text
    repaired = text
    for repair in KNOWN_TRANSCRIPT_REPAIRS:
        repaired = repaired.replace(repair.source, repair.target)
    return repaired


__all__ = [
    "KNOWN_TRANSCRIPT_REPAIRS",
    "KnownTranscriptRepair",
    "apply_known_transcript_repairs",
]
