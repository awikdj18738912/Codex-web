"""Quality-control checks for generated Refiner training pairs."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import LLM_MAX_TOKENS_DEFAULT
from src.services.llm_client import AsyncLLMClient, LLMClient
from src.pipeline.training_quality import AUDIT_SYSTEM, audit_pair
from src.utils.keyword_utils import dedupe_keywords, is_common_keyword, keyword_has_evidence

logger = logging.getLogger(__name__)

SEMANTIC_QC_SYSTEM = AUDIT_SYSTEM


class QCValidator:
    """Validate traceable error-prone keywords on generated records."""

    def __init__(self, async_client: AsyncLLMClient | None = None, max_concurrent: int = 8):
        self.llm = async_client or AsyncLLMClient(max_concurrent=max_concurrent)
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def validate_keywords_one(self, item: dict) -> tuple[bool, str, list[str]]:
        input_text = item.get("input", "")
        oral_text = item.get("oral_text", "")
        refined_text = item.get("output", {}).get("refined_text", "")
        keywords = dedupe_keywords(item.get("output", {}).get("keyword_list", []))
        if not keywords:
            return True, "no keywords", []

        common = [keyword for keyword in keywords if is_common_keyword(keyword)]
        if common:
            return False, f"keywords contain common terms: {common}", common

        source = "\n".join(part for part in (input_text, oral_text) if part)
        unsupported = [
            keyword
            for keyword in keywords
            if not keyword_has_evidence(keyword, source, refined_text)
        ]
        if unsupported:
            return False, f"keywords lack traceable evidence: {unsupported}", unsupported
        return True, "rule checks passed", []

    async def validate_batch(self, items: list[dict], lang: str = "zh") -> list[dict]:
        del lang
        valid: list[dict] = []
        for item in items:
            accepted, reason, _ = await self.validate_keywords_one(item)
            if accepted:
                valid.append(item)
            else:
                logger.debug("Keyword validation failed: %s", reason)
        return valid


async def validate_semantic_consistency(
    input_text: str,
    refined_text: str,
    llm: AsyncLLMClient,
) -> tuple[bool, str]:
    """Fail closed on incomplete cleanup, unsupported details or malformed audits."""
    passed, reason, _ = await audit_pair(input_text, refined_text, llm)
    return passed, reason


def check_itn_correctness(refined_text: str, original_text: str) -> tuple[bool, list[str]]:
    """Compatibility shim: number normalization is external; retained numbers stay verbatim.

    Keep this legacy check callable for existing pipeline integrations, but do not classify
    spoken-form numeric text as an error. Numeric surface consistency is validated elsewhere.
    """

    del refined_text, original_text
    return True, []


async def run_qc_on_dataset(items: list[dict], max_concurrent: int = 8) -> list[dict]:
    """Run keyword quality control over a dataset."""

    return await QCValidator(max_concurrent=max_concurrent).validate_batch(items)
