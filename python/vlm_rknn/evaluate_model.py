"""Run a prompt/gold-answer evaluation suite against an arbitrary model command.

The model command is run once per sample. The prompt is supplied on stdin unless
the command contains ``{prompt_file}``, in which case that placeholder is
replaced with the path to a temporary UTF-8 prompt file. Model stdout is treated
as the answer and stderr is retained in the JSON report.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


def normalize(text: str) -> str:
    """Normalize an answer for exact matching without changing its meaning."""
    return " ".join(text.casefold().split())


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)


def exact_match(actual: str, expected: str, _: dict[str, Any]) -> float:
    return float(normalize(actual) == normalize(expected))


def token_f1(actual: str, expected: str, _: dict[str, Any]) -> float:
    from collections import Counter

    actual_counts, expected_counts = Counter(tokens(actual)), Counter(tokens(expected))
    overlap = sum((actual_counts & expected_counts).values())
    if not actual_counts and not expected_counts:
        return 1.0
    if not actual_counts or not expected_counts or not overlap:
        return 0.0
    precision = overlap / sum(actual_counts.values())
    recall = overlap / sum(expected_counts.values())
    return 2 * precision * recall / (precision + recall)


def rouge_l(actual: str, expected: str, _: dict[str, Any]) -> float:
    a, b = tokens(actual), tokens(expected)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for index, right in enumerate(b, 1):
            current.append(previous[index - 1] + 1 if left == right else max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(a), lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def json_score(actual: str, expected: str, sample: dict[str, Any]) -> float:
    try:
        actual_value, expected_value = json.loads(actual), json.loads(expected)
    except json.JSONDecodeError:
        return 0.0
    required = sample.get("required_fields")
    if required:
        if not isinstance(actual_value, dict) or not isinstance(expected_value, dict):
            return 0.0
        matches = [field in actual_value and actual_value[field] == expected_value.get(field) for field in required]
        return sum(matches) / len(matches) if matches else 1.0
    return float(actual_value == expected_value)


EVALUATORS: dict[str, Callable[[str, str, dict[str, Any]], float]] = {
    "exact_match": exact_match,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
    "json": json_score,
}


@dataclass
class Result:
    id: str
    evaluator: str
    score: float
    passed: bool
    latency_seconds: float
    actual: str
    expected: str
    stderr: str
    returncode: int


def read_text(base: Path, value: str, field: str) -> str:
    path = (base / value).resolve()
    if not path.is_file():
        raise ValueError(f"{field} file does not exist: {path}")
    return path.read_text(encoding="utf-8")


def run_model(command: str, prompt: str, timeout: float) -> tuple[str, str, int, float]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt") as prompt_file:
        prompt_file.write(prompt)
        prompt_file.flush()
        argv = [part.replace("{prompt_file}", prompt_file.name) for part in shlex.split(command)]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                argv,
                input=None if "{prompt_file}" in command else prompt,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            elapsed = time.perf_counter() - started
            return completed.stdout.strip(), completed.stderr.strip(), completed.returncode, elapsed
        except subprocess.TimeoutExpired as error:
            elapsed = time.perf_counter() - started
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
            return stdout.strip(), (stderr + f"\nTimed out after {timeout:g}s").strip(), 124, elapsed


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Model evaluation",
        "",
        f"- Samples: {summary['samples']}",
        f"- Passed: {summary['passed']}",
        f"- Pass rate: {summary['pass_rate']:.1%}",
        f"- Mean score: {summary['mean_score']:.3f}",
        f"- Mean latency: {summary['mean_latency_seconds']:.3f} s",
        "",
        "| Sample | Evaluator | Score | Passed | Latency |",
        "| --- | --- | ---: | :---: | ---: |",
    ]
    for result in report["results"]:
        sample_id = str(result["id"]).replace("|", "\\|")
        lines.append(
            f"| {sample_id} | {result['evaluator']} | {result['score']:.3f} | "
            f"{'yes' if result['passed'] else 'no'} | {result['latency_seconds']:.3f} s |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON manifest containing a samples list")
    parser.add_argument("--command", required=True, help="Model command; reads stdin or use {prompt_file}")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation-results"))
    parser.add_argument("--timeout", type=float, default=120, help="Timeout per sample in seconds")
    parser.add_argument("--min-pass-rate", type=float, default=0.0)
    parser.add_argument("--min-mean-score", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest_path = args.manifest.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        samples = manifest["samples"]
        if not isinstance(samples, list) or not samples:
            raise ValueError("manifest 'samples' must be a non-empty list")
        if not 0 <= args.min_pass_rate <= 1 or not 0 <= args.min_mean_score <= 1:
            raise ValueError("quality thresholds must be between 0 and 1")

        results: list[Result] = []
        for index, sample in enumerate(samples, 1):
            sample_id = str(sample.get("id", index))
            evaluator_name = sample.get("evaluator", "exact_match")
            if evaluator_name not in EVALUATORS:
                raise ValueError(f"sample {sample_id}: unknown evaluator {evaluator_name!r}")
            prompt = read_text(manifest_path.parent, sample["prompt"], "prompt")
            expected = read_text(manifest_path.parent, sample["expected"], "expected").strip()
            actual, stderr, returncode, latency = run_model(args.command, prompt, args.timeout)
            score = EVALUATORS[evaluator_name](actual, expected, sample) if returncode == 0 else 0.0
            threshold = float(sample.get("threshold", 1.0 if evaluator_name in {"exact_match", "json"} else 0.8))
            results.append(Result(sample_id, evaluator_name, score, returncode == 0 and score >= threshold,
                                  latency, actual, expected, stderr, returncode))
            print(f"[{index}/{len(samples)}] {sample_id}: {score:.3f} {'PASS' if results[-1].passed else 'FAIL'}")

        passed = sum(result.passed for result in results)
        summary = {
            "samples": len(results),
            "passed": passed,
            "pass_rate": passed / len(results),
            "mean_score": sum(result.score for result in results) / len(results),
            "mean_latency_seconds": sum(result.latency_seconds for result in results) / len(results),
        }
        report = {"manifest": str(manifest_path), "command": args.command,
                  "summary": summary, "results": [asdict(result) for result in results]}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
        print(f"Reports written to {args.output_dir.resolve()}")
        return int(summary["pass_rate"] < args.min_pass_rate or summary["mean_score"] < args.min_mean_score)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
