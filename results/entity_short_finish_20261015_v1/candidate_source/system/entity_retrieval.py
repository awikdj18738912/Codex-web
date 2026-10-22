"""Broad entity recall for a trained verifier; this module never edits text."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Iterable

from rapidfuzz.distance import Levenshtein
from .entity_store import EntityDefinition
from .entity_matcher import _pinyin_tokens
from .entity_boundaries import candidate_boundary_issue


@dataclass(frozen=True, slots=True)
class RetrievedEntity:
    start: int
    end: int
    observed: str
    canonical: str
    entity_id: int
    entity_type: str
    phonetic_score: float
    character_score: float
    length_score: float
    retrieval_score: float
    boundary_issue: str | None

    def public_dict(self):
        from dataclasses import asdict
        return asdict(self)


@lru_cache(maxsize=65536)
def syllable_cost(left: str,right: str) -> float:
    if left==right:return 0.0
    distance=Levenshtein.normalized_distance(left,right)
    # Small segmental differences remain retrievable; the verifier decides
    # whether the occurrence means this entity. This is not confidence.
    return min(1.0,0.15+0.85*distance)


@lru_cache(maxsize=65536)
def _minimum_operation_cost(token: str, target: tuple[str,...]) -> float:
    return min(0.8,min(syllable_cost(token,other) for other in target))


@lru_cache(maxsize=131072)
def sound_similarity(left: tuple[str,...],right: tuple[str,...],minimum_similarity: float=0.0) -> float:
    if not left or not right:return 0.0
    # Relax alignment to arbitrary target positions for a lower bound on
    # edit cost. This can reject a pair, but cannot reject an above-cutoff
    # match. The audit's default still returns the exact similarity.
    if minimum_similarity:
        limit=(1-minimum_similarity)*max(len(left),len(right))
        if sum(_minimum_operation_cost(token,right) for token in left)>limit+1e-12:return 0.0
        if sum(_minimum_operation_cost(token,left) for token in right)>limit+1e-12:return 0.0
    previous=[j*0.8 for j in range(len(right)+1)]
    for i,a in enumerate(left,1):
        current=[i*0.8]
        for j,b in enumerate(right,1):
            current.append(min(previous[j]+0.8,current[-1]+0.8,previous[j-1]+syllable_cost(a,b)))
        previous=current
    return max(0.0,1.0-previous[-1]/max(len(left),len(right)))


class EntityRetriever:
    """Return per-span alternatives, retaining ambiguity and original offsets."""

    def __init__(self,entities: Iterable[EntityDefinition],*,minimum_score=0.52,max_candidates=5):
        self.minimum_score=minimum_score;self.max_candidates=max_candidates
        self.definitions={d.entity_id:d for d in entities if d.enabled and d.normalization_policy=='normalize'}
        self.by_length=defaultdict(list)
        self._surfaces={};self._syllable_index=defaultdict(set);self._near_cache={}
        for d in self.definitions.values():
            # Cross-script pronunciations must be separately sourced, not
            # invented as trusted aliases by this Chinese recall pass.
            if re.fullmatch(r'[\u3400-\u9fff]{2,24}',d.canonical_text):
                tokens=_pinyin_tokens(d.canonical_text)
                self.by_length[len(d.canonical_text)].append((d,tokens))
                self._surfaces[d.entity_id]=(d,tokens)
                for token in tokens:self._syllable_index[token].add(d.entity_id)

    def alternatives(self,text,start,end):
        """Score one supplied span, used for supervised training queries."""
        observed=text[start:end]
        if re.fullmatch(r'[\u3400-\u9fff]{2,26}',observed) is None:return ()
        source_py=_pinyin_tokens(observed);result=[]
        for size in range(max(2,len(observed)-2),min(24,len(observed)+2)+1):
            for d,target_py in self.by_length.get(size,()):
                if d.canonical_text==observed:continue
                phonetic=sound_similarity(source_py,target_py,0.55)
                if phonetic<0.55:continue
                chars=Levenshtein.normalized_similarity(observed,d.canonical_text)
                length=1-abs(len(observed)-size)/max(len(observed),size)
                score=0.8*phonetic+0.1*chars+0.1*length
                if score<self.minimum_score:continue
                result.append(RetrievedEntity(start,end,observed,d.canonical_text,d.entity_id,d.entity_type,
                    round(phonetic,6),round(chars,6),round(length,6),round(score,6),candidate_boundary_issue(text,start,end)))
        return tuple(sorted(result,key=lambda c:(-c.retrieval_score,c.entity_id))[:self.max_candidates])

    def retrieve(self,text: str,*,blocked_spans=()) -> tuple[RetrievedEntity,...]:
        by_span=defaultdict(list)
        for run in re.finditer(r'[\u3400-\u9fff]+',text):
            observed_run=run.group();pinyin=_pinyin_tokens(observed_run)
            near=[]
            for token in pinyin:
                if token not in self._near_cache:
                    self._near_cache[token]=set().union(*(ids for target,ids in self._syllable_index.items() if syllable_cost(token,target)<=0.5))
                near.append(self._near_cache[token])
            for i in range(len(observed_run)):
                for size in range(2,min(max(self.by_length,default=0)+2,len(observed_run)-i)+1):
                    start=run.start()+i;end=start+size
                    if any(start<b and a<end for a,b in blocked_spans):continue
                    observed=observed_run[i:i+size];source_py=pinyin[i:i+size]
                    pool=set().union(*near[i:i+size])
                    for entity_id in pool:
                        d,target_py=self._surfaces[entity_id]
                        target_size=len(d.canonical_text)
                        if abs(size-target_size)>2:continue
                        if observed==d.canonical_text:continue
                        # At least one matching/near syllable is required.
                        if not any(syllable_cost(a,b)<=0.5 for a in source_py for b in target_py):continue
                        phonetic=sound_similarity(source_py,target_py,0.55)
                        if phonetic<0.55:continue
                        chars=Levenshtein.normalized_similarity(observed,d.canonical_text)
                        length=1-abs(size-target_size)/max(size,target_size)
                        score=0.8*phonetic+0.1*chars+0.1*length
                        if score<self.minimum_score:continue
                        by_span[(start,end)].append(RetrievedEntity(start,end,observed,d.canonical_text,d.entity_id,
                            d.entity_type,round(phonetic,6),round(chars,6),round(length,6),round(score,6),
                            None))
        result=[]
        from dataclasses import replace
        for (start,end),alternatives in by_span.items():
            issue=candidate_boundary_issue(text,start,end)
            result.extend(replace(c,boundary_issue=issue) for c in sorted(alternatives,key=lambda c:(-c.retrieval_score,c.entity_id))[:self.max_candidates])
        return tuple(sorted(result,key=lambda c:(c.start,c.end,-c.retrieval_score,c.entity_id)))
