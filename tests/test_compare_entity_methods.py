"""Factorial scoring must not hide term swaps or ordinary-text regressions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
from compare_entity_methods import mention_changes,outside_entity_errors


def test_equal_total_with_different_names_is_both_gain_and_loss():
    assert mention_changes({'ICU':1,'MRSA':0},{'ICU':0,'MRSA':1})==(1,1)


def test_entity_omission_is_excluded_from_ordinary_text_error_count():
    reference='患者使用贝伐珠单抗有效'
    hypothesis='患者使用贝伐单抗有效'
    assert outside_entity_errors(reference,hypothesis,['贝伐珠单抗'])==(0,6)


def test_changed_negation_outside_entity_is_counted():
    reference='患者使用贝伐珠单抗有效'
    hypothesis='患者使用贝伐珠单抗无效'
    assert outside_entity_errors(reference,hypothesis,['贝伐珠单抗'])==(1,6)


def test_added_filler_after_entity_is_not_hidden_as_entity_error():
    assert outside_entity_errors('her2阳性','her2啊阳性',['HER2'])==(1,2)
