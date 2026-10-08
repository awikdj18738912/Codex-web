"""Temporary entity matcher hook for a real Web/GPU development acceptance."""
import time
from pathlib import Path
import system.entity_matcher as matching
from system.entity_matcher import EntityMatch,EntityMatchReport,EntityDecision,EntityFuzzyMode
from system.entity_pronunciation import pronunciation_signature
from probe_long_procedure_near import propose
import test_chinese_entity_heading_web as replay

replay.WEB=Path('results/entity_chinese_rules_20261008/long_near_web')
original_match=matching.EntityCandidateMatcher.match


def proposed_match(self,text,*,allow_auto,blocked_spans=()):
    blocked=tuple(blocked_spans);result=original_match(self,text,allow_auto=allow_auto,blocked_spans=blocked)
    if self.mode is not EntityFuzzyMode.AUTO or not allow_auto:return result
    started=time.perf_counter();_,edits=propose(text,tuple(self._entities.values()),True);matches=list(result.matches)
    for e in edits:
        a,b=e['start'],e['end']
        if any(a<y and x<b for x,y in blocked):continue
        surface=next(s for s in self._surfaces if s.value==e['canonical']);d=surface.definition
        score=self._score(text[a:b],pronunciation_signature(text[a:b]),surface)
        matches=[m for m in matches if not(a<m.end and m.start<b)]
        matches.append(EntityMatch(start=a,end=b,observed=text[a:b],canonical=d.canonical_text,
            entity_id=d.entity_id,entity_type=d.entity_type,domain=d.domain,matched_surface=d.canonical_text,
            match_type='CONTEXT_NEAR_PINYIN',char_score=score.char_score,pinyin_score=score.pinyin_score,
            length_score=score.length_score,domain_score=score.domain_score,priority_score=score.priority_score,
            final_score=score.final_score,runner_up_score=None,margin=None,priority=d.priority,
            decision=EntityDecision.AUTO_NORMALIZE,reasons=('experimental_long_procedure_action_near',)))
    return EntityMatchReport(tuple(sorted(matches,key=lambda m:m.start)),result.matcher_latency_ms+(time.perf_counter()-started)*1000)


def run():
    # The existing replay toggles its procedure verifier between these two
    # callables; intercept that assignment to toggle only our temporary hook.
    # A module __class__ setter is local to this experimental process.
    import types
    original_module_class=matching.__class__
    base_verifier=matching.procedure_spelling_evidence
    class Toggled(types.ModuleType):
        def __setattr__(self,name,value):
            if name=='procedure_spelling_evidence':
                matching.EntityCandidateMatcher.match=original_match if value is base_verifier else proposed_match
                value=base_verifier
            super().__setattr__(name,value)
    matching.__class__=Toggled
    try:replay.run()
    finally:
        matching.__class__=original_module_class
        matching.EntityCandidateMatcher.match=original_match


if __name__=='__main__':run()
