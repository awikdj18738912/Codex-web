"""Compare follow-up performance trials against exact baseline transcripts."""

import argparse
import json
from pathlib import Path
import statistics

from report_speed_quality import differences


def summarize(root: Path):
    groups = {
        "baseline": ("baseline_a1", "baseline_a2"),
        "cpu_threads": ("threads_b1", "threads_b2"),
        "tf32": ("tf32_c1", "tf32_c2"),
        "restored": ("restored_d1",),
    }
    rows = []
    for audio in (4, 5):
        baseline = json.loads((root / f"baseline_a1_测试音频{audio}.json").read_text())
        baseline_hashes = baseline["_speed_benchmark"]["configuration"]["hashes"]
        for group, labels in groups.items():
            runs = []
            for label in labels:
                path = root / f"{label}_测试音频{audio}.json"
                if not path.exists():
                    continue
                result = json.loads(path.read_text())
                metrics = result["_speed_benchmark"]
                stats = result["refiner_session_stats"]
                runs.append({
                    "label": label,
                    "elapsed_s": metrics["elapsed_s"],
                    "feed_s": metrics["feed_s"],
                    "finish_s": metrics["finish_s"],
                    "model_s": metrics["model_sum_s"],
                    "refiner_wall_s": stats["completed_wall_latency_ms"] / 1000,
                    "refiner_calls": stats["call_count"],
                    "raw_equal": result["raw_text"] == baseline["raw_text"],
                    "clean_equal": result["clean_text"] == baseline["clean_text"],
                    "raw_differences": differences(baseline["raw_text"], result["raw_text"]),
                    "clean_differences": differences(baseline["clean_text"], result["clean_text"]),
                    "source_hashes_unchanged": metrics["configuration"]["hashes"] == baseline_hashes,
                })
            if runs:
                rows.append({
                    "audio": audio, "variant": group, "runs": runs,
                    "mean_elapsed_s": statistics.mean(run["elapsed_s"] for run in runs),
                    "all_raw_equal": all(run["raw_equal"] for run in runs),
                    "all_clean_equal": all(run["clean_equal"] for run in runs),
                    "all_source_hashes_unchanged": all(run["source_hashes_unchanged"] for run in runs),
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    root = parser.parse_args().directory
    rows = summarize(root)
    (root / "comparison.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n"
    )
    names = {"baseline": "原配置", "cpu_threads": "CPU 单线程", "tf32": "TF32", "restored": "恢复验证"}
    lines = ["# 后续速度试验", "", "采用完整离线流式路径，暖机不计入耗时。", "",
             "| 音频 | 方案 | 轮数 | 总耗时均值（秒） | 原始文字一致 | 精修逐字一致 | 源码和规则库哈希一致 |",
             "| --- | --- | ---: | ---: | --- | --- | --- |"]
    for row in rows:
        same = lambda key: "是" if row[key] else "否"
        lines.append(f"| {row['audio']} | {names[row['variant']]} | {len(row['runs'])} | "
                     f"{row['mean_elapsed_s']:.3f} | {same('all_raw_equal')} | "
                     f"{same('all_clean_equal')} | {same('all_source_hashes_unchanged')} |")
        print({key: value for key, value in row.items() if key != "runs"})
    (root / "comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
