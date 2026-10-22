from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from contextlib import nullcontext

from system.refinement_protocol import (
    apply_window_context_patch,
    window_context_messages,
)
from system.live_qwen_refiner import TransformersRefiner
from system.refinement_guard import window_context_reject_reasons


class WindowContextProtocolTest(unittest.TestCase):
    def test_window_context_json_round_trips_special_characters_and_newlines(self):
        source = '当前窗口：他说「A\\B」\n第二行\t“引号”'
        left = '前文有 emoji 😀、\n以及"双引号"'
        right = "后文含反斜杠 \\\n 继续"

        suffix, content = window_context_messages(
            source, left_context=left, right_context=right
        )

        self.assertIsInstance(suffix, str)
        self.assertEqual(content, source)
        context_json = suffix.split("只读上下文JSON：", 1)[1]
        self.assertEqual(
            json.loads(context_json),
            {"read_only_left_context": left, "read_only_right_context": right},
        )

    def test_suffix_limits_edits_to_current_window_and_retains_subject_context(self):
        suffix, content = window_context_messages(
            "他后来到了那里。", left_context="小王先去了医院，", right_context="医生说他已经好了。"
        )
        self.assertEqual(content, "他后来到了那里。")
        self.assertIn("唯一允许修改和输出的正文", suffix)
        self.assertIn("不得修改、复制到输出或作为补丁source", suffix)
        self.assertTrue(any(term in suffix for term in ("主语", "指代", "人物")))
        self.assertTrue(any(term in suffix for term in ("保留", "原样")))

    def test_window_patch_rejects_malformed_and_duplicate_json(self):
        source = "他后来到了那里。"
        malformed = '{"action":"replace","source":"那里",'
        duplicate_field = (
            '{"action":"replace","source":"那里","source":"后来",'
            '"target":"医院","reason":"local_typo"}'
        )
        for response in (malformed, duplicate_field):
            with self.subTest(response=response):
                refined, _payload, issue = apply_window_context_patch(source, response)
                self.assertEqual(refined, source)
                self.assertEqual(issue, "window_context_invalid_patch")

    def test_compat_prose_is_parsed_then_guard_rejects_copied_context(self):
        source = "他后来到了那里。"
        left = "小王先去了医院，"
        right = "医生说他已经好了。"
        whole_text = left + source + right

        refined, payload, issue = apply_window_context_patch(source, whole_text)

        self.assertEqual(refined, whole_text)
        self.assertIsNone(payload)
        self.assertIsNone(issue)
        self.assertIn(
            "window_context_copied",
            window_context_reject_reasons(
                source, refined, left_context=left, right_context=right
            ),
        )

    def test_strict_local_context_patch_rejects_prose_output(self):
        source = "他后来到了那里。"
        refined, _payload, issue = apply_window_context_patch(
            source, "他后来到了医院。", policy="strict_local"
        )
        self.assertEqual(refined, source)
        self.assertEqual(issue, "window_context_invalid_patch")

    def test_window_patch_accepts_only_a_local_current_window_patch(self):
        source = "他后来到了那里。"
        response = (
            '{"action":"replace","source":"那里","target":"医院",'
            '"reason":"local_typo"}'
        )
        refined, payload, issue = apply_window_context_patch(source, response)
        self.assertEqual(refined, "他后来到了医院。")
        self.assertEqual(payload["source"], "那里")
        self.assertIsNone(issue)


class TransformersWindowContextTest(unittest.TestCase):
    def test_refine_apis_forward_context_and_keep_their_policy(self):
        refiner = object.__new__(TransformersRefiner)
        refiner.window_edit_policy = "strict_local"
        context = {
            "left_context": "前一窗",
            "right_context": "后一窗",
        }
        calls = []

        def capture(*args, **kwargs):
            calls.append((args, kwargs))
            return "result", 1.0

        with patch.object(refiner, "_refine_with_system_prompt", side_effect=capture):
            self.assertEqual(refiner.refine("当前", window_context=context), ("result", 1.0))
            self.assertEqual(
                refiner.refine_with_numeric_rules("当前", window_context=context),
                ("result", 1.0),
            )

        self.assertEqual([call[1]["window_context"] for call in calls], [context, context])
        self.assertIn("严格JSON", calls[0][0][1])
        self.assertIn("数字、单位、范围", calls[1][0][1])

    def test_generated_messages_keep_hints_and_candidates_outside_window_json(self):
        refiner = object.__new__(TransformersRefiner)
        captured = []

        class Inputs(dict):
            class Shape:
                def __getitem__(self, _index):
                    return 1

            def to(self, _device):
                return self

            def __getitem__(self, key):
                if key == "input_ids":
                    return type("InputIds", (), {"shape": self.Shape()})()
                return super().__getitem__(key)

        class Tokenizer:
            def apply_chat_template(self, messages, **_kwargs):
                captured.extend(messages)
                return Inputs(input_ids=[[0]])

        class Embeddings:
            weight = type("Weight", (), {"device": "cpu"})()

        class Model:
            def get_input_embeddings(self):
                return Embeddings()

        refiner._tokenizer = Tokenizer()
        refiner._model = Model()
        refiner._torch = type("Torch", (), {"inference_mode": staticmethod(nullcontext)})
        refiner._template_kwargs = {}
        context = {
            "left_context": "前文",
            "right_context": "后文",
        }
        candidates = [{"observed": "甲", "canonical": "乙"}]
        with patch.object(refiner, "_generate", return_value=object()), patch.object(
            refiner, "_decode_generated", return_value=""
        ):
            refiner._refine_with_system_prompt(
                "窗口文本",
                "base prompt",
                entity_hints=("提示词",),
                strict_placeholders=False,
                entity_candidates=candidates,
                window_context=context,
            )

        system_message, user_message = captured
        self.assertIn("用户消息是当前窗口", system_message["content"])
        content = user_message["content"]
        source_and_hints, candidate_trailing = content.split("\n<ENTITY_CANDIDATES>", 1)
        self.assertEqual(source_and_hints, "窗口文本\n<KEY>[提示词]")
        self.assertEqual(json.loads(candidate_trailing), candidates)
        self.assertIn("只读上下文JSON：", system_message["content"])
        context_json = system_message["content"].split("只读上下文JSON：", 1)[1]
        context_payload, _end = json.JSONDecoder().raw_decode(context_json)
        self.assertEqual(
            context_payload,
            {"read_only_left_context": "前文", "read_only_right_context": "后文"},
        )


if __name__ == "__main__":
    unittest.main()
