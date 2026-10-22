"""Check current QC on known bad v1 labels and valid counterparts."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
os.environ.update(VLLM_BASE_URL="http://127.0.0.1:8010/v1", VLLM_MODEL_NAME="refiner-teacher",
                  OPENROUTER_API_KEY="", OPENROUTER_MODEL="")
sys.path.insert(0, str(ROOT / "pipeline"))
from src.services.llm_client import AsyncLLMClient

spec = importlib.util.spec_from_file_location("current_finalize", ROOT / "pipeline/scripts/05_finalize.py")
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)


async def main():
    prior = {row["id"]: row for row in json.loads((OUT.parent / "refiner_teacher_pilot_20261017/fixed_checks.json").read_text())}
    correct = {
        "fixed_numeric_correction": "预算改成500元。",
        "fixed_rollback": "我想用红色吧。",
        "fixed_multiple": "最后定周三开会。",
        "fixed_spelling_cue": "收件人是钟宁。",
    }
    # Source uses the same numeric normalization as the actual generation path.
    from src.utils.numeric_policy import normalize_source
    client = AsyncLLMClient(max_concurrent=2, timeout=600, max_retries=1)
    sem = asyncio.Semaphore(2)

    async def one(case_id, label_kind, target):
        source, _ = normalize_source(prior[case_id]["source"])
        item = {"id": case_id + "_" + label_kind, "input": source,
                "output": {"refined_text": target}}
        started = time.monotonic()
        _, passed, reason = await qc.qc_one(item, client, sem)
        return {"id": item["id"], "case": case_id, "label_kind": label_kind,
                "source": source, "target": target, "expected_pass": label_kind == "correct",
                "actual_pass": bool(passed), "verdict_correct": bool(passed) == (label_kind == "correct"),
                "reason": reason, "elapsed_seconds": round(time.monotonic() - started, 3)}

    rows = await asyncio.gather(*(one(case_id, kind, target)
        for case_id, good in correct.items()
        for kind, target in (("known_bad_v1", prior[case_id]["target"]), ("correct", good))))
    result = {"total": len(rows), "verdict_correct": sum(row["verdict_correct"] for row in rows),
              "known_bad_rejected": sum(not row["actual_pass"] for row in rows if row["label_kind"] == "known_bad_v1"),
              "correct_accepted": sum(row["actual_pass"] for row in rows if row["label_kind"] == "correct"),
              "note": "Diagnostic controls call semantic QC directly; numeric/entity admission is evaluated separately.",
              "items": rows}
    (OUT / "qc_controls.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    await client.client.close()


if __name__ == "__main__":
    asyncio.run(main())
