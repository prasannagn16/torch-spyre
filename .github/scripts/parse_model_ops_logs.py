#!/usr/bin/env python3
"""
Parse raw GHA job logs produced by the model-ops-tests workflow and produce:

  1. model_ops_log.txt  – cleaned, concatenated plain-text log (one file per run)
  2. <out>.json         – structured per-variant records ready for ClickHouse
                          ingest (see ingest_model_ops.py)

Each GHA job log file corresponds to one model suite (e.g. "GPT OSS 20B Spyre").
The parser extracts every individual pytest test-case result (XPASS / XFAIL /
FallbackWarning) using the same logic as run_all_test.py::TestLogAnalyzer,
producing one JSON record per variant.

JSON output schema (array of objects)
--------------------------------------
  run_id              : GHA run ID string
  suite_name          : human-readable model suite (from filename)
  model_name          : config stem, e.g. "gpt-oss-20b"
  yaml_file           : config filename, e.g. "gpt-oss-20b_spyre.yaml"
  operation           : torch op name, e.g. "torch.mul"
  classification      : "spyre_enabled" | "not_implemented" | "cpu_fallback"
  test_name           : pytest node id, e.g. "test_model_ops_db_torch_mul__1_spyre_float16"
  status              : "XPASS" | "XFAIL" | "FALLBACK"
  input_shapes        : list[str] of per-tensor shape strings, e.g. ["[1,12,4096]"]
  input_strides       : list[str] of per-tensor stride strings
  input_dtypes        : list[str] of per-tensor dtype strings, e.g. ["torch.float16"]
  arg_values          : list[str] of non-tensor argument values (may be empty)
  target_shape        : reshaped target shape string (for reshape/view ops, else "")
  triggered_at        : ISO-8601 timestamp of first GHA log line in this file
  ingested_at         : ISO-8601 timestamp (now)

  # Suite-level fields (same for every variant in a suite)
  suite_outcome       : "passed" | "failed" | "error" | "unknown"
  suite_exit_code     : int or null
  suite_tests_total   : int
  suite_tests_passed  : int
  suite_tests_failed  : int
  suite_tests_skipped : int
  suite_tests_error   : int
  suite_tests_xfail   : int
  suite_tests_xpass   : int
  suite_duration_s    : float

Usage
-----
  python3 parse_model_ops_logs.py \\
      --log-dir  raw_logs/ \\
      --run-id   <GHA_RUN_ID> \\
      --out      model_ops_<run_id>.json \\
      --log-out  model_ops_log.txt
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# GHA / pytest line patterns
# ---------------------------------------------------------------------------

# GHA step timestamp prefix:  2025-01-15T10:23:45.1234567Z  text…
RE_GHA_TS = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+(?P<rest>.*)$"
)

# Pytest summary line detector — presence only (last line with "= ... in Xs =" wins)
# Example: "================== 5 passed, 1 xfailed, 2 xpassed in 9.00s =================="
RE_PYTEST_SUMMARY = re.compile(r"={3,}.*\bin (?P<secs>[\d.]+)s")

# Individual count patterns extracted separately from a summary line
_RE_SUMM_FAIL = re.compile(r"(\d+) failed")
_RE_SUMM_PASS = re.compile(r"(\d+) passed")
_RE_SUMM_SKIP = re.compile(r"(\d+) skipped")
_RE_SUMM_ERR = re.compile(r"(\d+) error")
_RE_SUMM_XFAIL = re.compile(r"(\d+) xfailed")
_RE_SUMM_XPASS = re.compile(r"(\d+) xpassed")
RE_COLLECTED = re.compile(r"collected (?P<n>\d+) item")
RE_GHA_EXIT = re.compile(
    r"Error: Process completed with exit code (?P<code>\d+)", re.IGNORECASE
)
RE_COLLECT_ERROR = re.compile(r"ERROR collecting", re.IGNORECASE)
RE_TIMEOUT = re.compile(
    r"(The job running on runner .+ exceeded the maximum execution time"
    r"|No new output for \d+s.*stall)",
    re.IGNORECASE,
)

# ANSI / control chars
_RE_ANSI = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")

# ---------------------------------------------------------------------------
# TestLogAnalyzer patterns  (ported from run_all_test.py)
# ---------------------------------------------------------------------------

# XPASS / XFAIL status lines emitted by pytest
RE_XPASS_LINE = re.compile(r"^\s*XPASS")
RE_XFAIL_LINE = re.compile(r"^\s*XFAIL")

# Pytest verbose separator line:
#   "test_model_ops_v2.py::TestSpyreModelOpsPRIVATEUSE1::test_model_ops_db_torch_mul__1_spyre_float16"
RE_TEST_SEP = re.compile(
    r"(?:test_model_ops_v2\.py::[^:]+::|::|^)"
    r"(?P<test>test_model_ops_db_[\w]+)"
)

# Op info line:  "Op: torch.mul | Test: test_model_ops_db_torch_mul__1_spyre_float16"
RE_OP_LINE = re.compile(
    r"Op:\s+(?P<op>[\w.]+)\s+\|\s+Test:\s+(?P<test>test_model_ops_db_[\w]+)"
)

# Single-tensor input:
#   "Input: shape=[1, 41, 4096], stride=[167936, 4096, 1], dtype=torch.bfloat16"
RE_INPUT_SINGLE = re.compile(
    r"Input:\s+shape=(?P<shape>\[[\d,\s]+\]),\s+"
    r"stride=(?P<stride>\[[\d,\s]+\]),\s+"
    r"dtype=(?P<dtype>torch\.\w+)"
)

# Tensor in a list:
#   "  [0]: shape=[1, 41, 64], stride=[2624, 1, 41], dtype=torch.float32"
# NOTE: after GHA timestamp prefix is stripped and _clean() is called the
# leading spaces are gone, so we match \s* (zero or more) not \s+.
RE_INPUT_LIST_ITEM = re.compile(
    r"^\s*\[\d+\]:\s+shape=(?P<shape>\[[\d,\s]+\]),\s+"
    r"stride=(?P<stride>\[[\d,\s]+\]),\s+"
    r"dtype=(?P<dtype>torch\.\w+)"
)

# Tensor in Args block:
#   "  [0]: Tensor(shape=[1, 41, 4096], stride=[167936, 4096, 1], dtype=torch.bfloat16)"
RE_ARGS_TENSOR = re.compile(
    r"^\s*\[\d+\]:\s+Tensor\(shape=(?P<shape>\[[\d,\s]+\]),\s+"
    r"stride=(?P<stride>\[[\d,\s]+\]),\s+"
    r"dtype=(?P<dtype>torch\.\w+)\)"
)

# Non-tensor arg value lines:  "  [0]: 1e-05"  or  "  [1]: 'torch.float32'"
RE_ARG_VALUE = re.compile(r"^\s*\[\d+\]:\s+(?!Tensor\()(?P<val>.+)$")

# Target shape lines (for reshape/view):  "Target shape: (1, 12, -1, 128)"
RE_TARGET_SHAPE = re.compile(r"Target shape:\s+(?P<shape>\([^)]+\))")

# FallbackWarning:
#   "FallbackWarning: aten.cos.default is falling back to cpu"
RE_FALLBACK_ATEN = re.compile(
    r"FallbackWarning:\s+(?P<op>aten\.[\w.]+)\s+is falling back"
)
#   "FallbackWarning: conversion from torch.int64 to torch.float32 is falling back"
RE_FALLBACK_CONV = re.compile(
    r"FallbackWarning:\s+conversion from\s+(?P<src>torch\.\w+)\s+to\s+"
    r"(?P<dst>torch\.\w+)\s+is falling back"
)

# Input: List of N tensors  (header — no data, just marks start)
RE_INPUT_LIST_HDR = re.compile(r"Input:\s+List of \d+ tensors:")


def _clean(s: str) -> str:
    s = _RE_ANSI.sub("", s)
    s = _RE_CTRL.sub("", s)
    return s.strip()


def _strip_gha_prefix(line: str):
    """Return (iso_ts_or_None, bare_line)."""
    m = RE_GHA_TS.match(line)
    if m:
        return m.group("ts"), m.group("rest")
    return None, line


def _parse_ts(ts_str):
    if not ts_str:
        return datetime.now(timezone.utc).isoformat()
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return datetime.now(timezone.utc).isoformat()


def _compact_shape(raw: str) -> str:
    """Normalise shape string → '[1,12,4096]' (no spaces)."""
    return re.sub(r"\s+", "", raw)


# ---------------------------------------------------------------------------
# Suite-name / model-name extraction from filename
# ---------------------------------------------------------------------------

_SKIP_NAMES = re.compile(
    r"^(detect changed|run spyre unit|ingest|push.*(clickhouse|diagnostics)|"
    r"checkout|install|derive|upload|build|gather|set up)",
    re.IGNORECASE,
)


def _suite_from_filename(filename: str):
    """Return (suite_name_or_None, model_name_or_None)."""
    stem = re.sub(r"\.txt$", "", filename, flags=re.IGNORECASE)
    stem = re.sub(r"^\d+_", "", stem).strip()
    if _SKIP_NAMES.match(stem):
        return None, None

    # Strip "run-tests _ " prefix if present
    m = re.match(r"^run-tests\s*[_|]\s*(.+)$", stem, re.IGNORECASE)
    if m:
        suite_name = m.group(1).strip()
    elif re.search(r"[A-Z]", stem) and (" " in stem or "-" in stem):
        suite_name = stem
    else:
        return None, None

    # Derive model_name: lower-case, spaces→hyphens, collapse repeated hyphens
    model_name = re.sub(r"\s+", "-", suite_name.lower())
    model_name = re.sub(r"-+", "-", model_name).strip("-")

    return suite_name, model_name


def _yaml_file_from_model_name(model_name: str) -> str:
    """Best-effort guess at the yaml filename."""
    # e.g. "gpt-oss-20b-spyre" → "gpt-oss-20b_spyre.yaml"
    # Suite names ending in " Spyre" map to _spyre.yaml
    slug = re.sub(r"-spyre$", "_spyre", model_name)
    return f"{slug}.yaml"


def _pick_log_files(log_dir: Path):
    """Return (path, suite_name, model_name) for every recognisable file."""
    candidates = []
    for fpath in sorted(log_dir.iterdir()):
        if not fpath.is_file() or fpath.name.startswith("."):
            continue
        if fpath.suffix not in (".txt", ".log", ""):
            continue
        suite, model = _suite_from_filename(fpath.name)
        if suite is None:
            continue
        candidates.append((fpath, suite, model))

    # Deduplicate: prefer .txt over extension-less
    seen: dict = {}
    for fpath, suite, model in candidates:
        if suite not in seen:
            seen[suite] = (fpath, model)
        elif fpath.suffix in (".txt", ".log") and seen[suite][0].suffix == "":
            seen[suite] = (fpath, model)

    return [(path, suite, model) for suite, (path, model) in sorted(seen.items())]


# ---------------------------------------------------------------------------
# Core per-file parser  (TestLogAnalyzer logic)
# ---------------------------------------------------------------------------


class _TestLogAnalyzer:
    """
    Stateful line-by-line parser for a single model-ops GHA job log.
    Mirrors the run_all_test.py TestLogAnalyzer logic exactly.
    """

    def __init__(self):
        # Variant storage: key → dict
        self.xpass_variants: dict[str, dict] = {}
        self.xfail_variants: dict[str, dict] = {}
        self.fallback_ops: set = set()

        # Per-test state (reset on each new test separator line)
        self._test_name: str | None = None
        self._op_name: str | None = None
        self._shapes: list[str] = []
        self._strides: list[str] = []
        self._dtypes: list[str] = []
        self._args: list[str] = []
        self._target_shape: str = ""
        self._in_block: bool = False

    # ---- public ---------------------------------------------------------
    #
    # Real pytest -v -s log order for each test variant:
    #
    #   test_model_ops_v2.py::Class::test_name   ← separator (reset state)
    #     Op: torch.mul | Test: test_name         ← set op_name + test_name
    #     Input: shape=…, stride=…, dtype=…       ← collect tensor info
    #     Args:
    #       [0]: 0.5                               ← collect arg values
    #     Target shape: (…)                        ← optional for reshape
    #   XPASS […]                                  ← COMMIT (status known now)
    #
    # This matches run_all_test.py::TestLogAnalyzer.parse_log_line() exactly.

    def feed(self, line: str):
        line = _clean(line)

        # ── FallbackWarning (can appear anywhere) ────────────────────────
        if "FallbackWarning" in line:
            m = RE_FALLBACK_ATEN.search(line)
            if m:
                aten = m.group("op")
                torch_op = aten.replace("aten.", "torch.").split(".default")[0]
                self.fallback_ops.add(torch_op)
                return
            m = RE_FALLBACK_CONV.search(line)
            if m:
                self.fallback_ops.add(
                    f"type_conversion_{m.group('src')}_to_{m.group('dst')}"
                )
                return

        # ── Pytest verbose separator → start new test block ──────────────
        # Must be checked BEFORE Op: line so we always reset on a new test.
        m = RE_TEST_SEP.search(line)
        if m and "::" in line:
            self._start_new_test(m.group("test"))
            return

        # ── Op info line → record op name + test name ────────────────────
        m = RE_OP_LINE.search(line)
        if m:
            self._op_name = m.group("op")
            self._test_name = m.group("test")
            self._in_block = True
            return

        # ── XPASS / XFAIL → COMMIT (data already collected above) ────────
        # These appear AFTER Op:/Input:/Args: lines in real pytest output.
        if RE_XPASS_LINE.match(line):
            if self._op_name and self._test_name:
                self._commit("XPASS")
            return
        if RE_XFAIL_LINE.match(line):
            if self._op_name and self._test_name:
                self._commit("XFAIL")
            return

        if not self._in_block:
            return

        # ── Target shape ─────────────────────────────────────────────────
        m = RE_TARGET_SHAPE.search(line)
        if m:
            self._target_shape = m.group("shape")
            return

        # ── Single-tensor input ──────────────────────────────────────────
        m = RE_INPUT_SINGLE.search(line)
        if m:
            self._shapes.append(_compact_shape(m.group("shape")))
            self._strides.append(_compact_shape(m.group("stride")))
            self._dtypes.append(m.group("dtype"))
            return

        # ── List-of-tensors header ────────────────────────────────────────
        if RE_INPUT_LIST_HDR.search(line):
            return  # header only; items follow on next lines

        # ── Tensor within a list ─────────────────────────────────────────
        m = RE_INPUT_LIST_ITEM.search(line)
        if m:
            self._shapes.append(_compact_shape(m.group("shape")))
            self._strides.append(_compact_shape(m.group("stride")))
            self._dtypes.append(m.group("dtype"))
            return

        # ── Tensor in Args block ─────────────────────────────────────────
        m = RE_ARGS_TENSOR.search(line)
        if m:
            self._shapes.append(_compact_shape(m.group("shape")))
            self._strides.append(_compact_shape(m.group("stride")))
            self._dtypes.append(m.group("dtype"))
            return

        # ── Non-tensor arg value ─────────────────────────────────────────
        m = RE_ARG_VALUE.match(line)
        if m:
            val = m.group("val").strip()
            # Skip lines that look like shape/stride/dtype summaries
            if not re.match(r"^(shape|stride|dtype|Tensor|Args)\b", val, re.IGNORECASE):
                self._args.append(val)

    def finish(self):
        """No-op — in real logs every test ends with XPASS/XFAIL before the
        next separator, so nothing is left pending. Kept for safety."""
        pass

    # ---- internal -------------------------------------------------------

    def _start_new_test(self, test_name: str):
        """Reset all per-test state for a new test separator line."""
        self._op_name = None
        self._test_name = test_name
        self._shapes = []
        self._strides = []
        self._dtypes = []
        self._args = []
        self._target_shape = ""
        self._in_block = True

    def _commit(self, status: str):
        if not (self._op_name and self._test_name):
            return
        key = f"{self._op_name}|{self._test_name}"
        record = {
            "operation": self._op_name,
            "classification": "spyre_enabled"
            if status == "XPASS"
            else "not_implemented",
            "test_name": self._test_name,
            "input_shapes": list(self._shapes),
            "input_strides": list(self._strides),
            "input_dtypes": list(self._dtypes),
            "arg_values": list(self._args),
            "target_shape": self._target_shape,
            "status": status,
        }
        if status == "XPASS":
            self.xpass_variants[key] = record
        else:
            self.xfail_variants[key] = record
        self._reset()

    def _reset(self):
        self._op_name = None
        self._test_name = None
        self._shapes = []
        self._strides = []
        self._dtypes = []
        self._args = []
        self._target_shape = ""
        self._in_block = False


def _group_by_operation(variants: list[dict]) -> list[dict]:
    """
    Group a flat list of per-variant dicts by operation name,
    matching the output schema shown in the dashboard JSON.
    """
    grouped: dict[str, list] = defaultdict(list)
    for v in variants:
        grouped[v["operation"]].append(v)
    result = []
    for op_name in sorted(grouped):
        group_variants = grouped[op_name]
        result.append(
            {
                "operation": op_name,
                "variant_count": len(group_variants),
                "variants": group_variants,
            }
        )
    return result


def _build_spyre_failed(
    xpass_variants: list[dict], xfail_variants: list[dict]
) -> list[dict]:
    """
    Build the 'spyre_failed' section: operations that have BOTH xpass AND xfail
    variants (partial support — some shapes work, some don't).
    """
    xpass_by_op: dict[str, list] = defaultdict(list)
    xfail_by_op: dict[str, list] = defaultdict(list)
    for v in xpass_variants:
        xpass_by_op[v["operation"]].append(v)
    for v in xfail_variants:
        xfail_by_op[v["operation"]].append(v)

    mixed_ops = set(xpass_by_op) & set(xfail_by_op)
    result = []
    for op_name in sorted(mixed_ops):
        xp = xpass_by_op[op_name]
        xf = xfail_by_op[op_name]
        result.append(
            {
                "operation": op_name,
                "xpass_count": len(xp),
                "xfail_count": len(xf),
                "xpass_variants": xp,
                "xfail_variants": xf,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Suite-level stats from log text
# ---------------------------------------------------------------------------


def _parse_suite_stats(lines: list[str]) -> dict:
    stats = {
        "outcome": "unknown",
        "exit_code": None,
        "tests_total": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0,
        "tests_error": 0,
        "tests_xfail": 0,
        "tests_xpass": 0,
        "duration_s": 0.0,
    }
    chunk = "\n".join(lines)

    m_col = RE_COLLECTED.search(chunk)
    if m_col:
        stats["tests_total"] = int(m_col.group("n"))

    # Parse last pytest summary line using individual patterns
    def _first_int(pattern, text, default=0):
        mm = pattern.search(text)
        return int(mm.group(1)) if mm else default

    for line in reversed(lines):
        m = RE_PYTEST_SUMMARY.search(line)
        if m:
            stats["tests_failed"] = _first_int(_RE_SUMM_FAIL, line)
            stats["tests_passed"] = _first_int(_RE_SUMM_PASS, line)
            stats["tests_skipped"] = _first_int(_RE_SUMM_SKIP, line)
            stats["tests_error"] = _first_int(_RE_SUMM_ERR, line)
            stats["tests_xfail"] = _first_int(_RE_SUMM_XFAIL, line)
            stats["tests_xpass"] = _first_int(_RE_SUMM_XPASS, line)
            try:
                stats["duration_s"] = float(m.group("secs"))
            except (TypeError, ValueError):
                pass
            if stats["tests_total"] == 0:
                stats["tests_total"] = (
                    stats["tests_passed"]
                    + stats["tests_failed"]
                    + stats["tests_skipped"]
                    + stats["tests_error"]
                )
            break

    # Outcome
    m_exit = RE_GHA_EXIT.search(chunk)
    if m_exit:
        stats["exit_code"] = int(m_exit.group("code"))
        stats["outcome"] = "failed" if stats["exit_code"] != 0 else "passed"
    elif stats["tests_failed"] > 0 or stats["tests_error"] > 0:
        stats["outcome"] = "failed"
        stats["exit_code"] = 1
    elif stats["tests_passed"] > 0 or stats["tests_xpass"] > 0:
        stats["outcome"] = "passed"
        stats["exit_code"] = 0
    elif RE_COLLECT_ERROR.search(chunk):
        stats["outcome"] = "error"
        stats["exit_code"] = 2

    return stats


# ---------------------------------------------------------------------------
# Per-file entry point
# ---------------------------------------------------------------------------


def parse_log_file(
    text: str,
    run_id: str,
    suite_name: str,
    model_name: str,
) -> dict:
    """
    Parse one GHA job log and return a single suite-level record that contains
    every per-variant result plus the suite summary stats.
    """
    raw_lines = text.splitlines()

    # Strip GHA timestamps, collect first timestamp
    lines: list[str] = []
    first_ts: str | None = None
    for raw in raw_lines:
        ts, bare = _strip_gha_prefix(raw)
        if ts and first_ts is None:
            first_ts = ts
        lines.append(_clean(bare))

    # Suite-level stats
    stats = _parse_suite_stats(lines)

    # Per-variant analysis
    analyzer = _TestLogAnalyzer()
    for line in lines:
        analyzer.feed(line)
    analyzer.finish()

    xpass_list = list(analyzer.xpass_variants.values())
    xfail_list = list(analyzer.xfail_variants.values())

    # Build CPU-fallback list — pure fallback ops (not XPASS/XFAIL)
    xpass_ops = {v["operation"] for v in xpass_list}
    xfail_ops = {v["operation"] for v in xfail_list}
    cpu_fallback_ops = analyzer.fallback_ops - xpass_ops - xfail_ops

    # Derive spyre-failed (mixed): ops with both XPASS and XFAIL variants
    # Remove those ops from the clean spyre_enabled / not_implemented groups
    mixed_ops = {v["operation"] for v in xpass_list} & {
        v["operation"] for v in xfail_list
    }
    pure_xpass = [v for v in xpass_list if v["operation"] not in mixed_ops]
    pure_xfail = [v for v in xfail_list if v["operation"] not in mixed_ops]

    yaml_file = _yaml_file_from_model_name(model_name)

    return {
        "run_id": run_id,
        "suite_name": suite_name,
        "model_name": model_name,
        "yaml_file": yaml_file,
        # Suite summary
        "summary": {
            "total_tests": stats["tests_total"],
            "spyre_enabled_count": len(pure_xpass),
            "not_implemented_count": len(pure_xfail),
            "cpu_fallback_count": len(cpu_fallback_ops),
            "spyre_failed_count": len(mixed_ops),
        },
        # Operations breakdown
        "operations": {
            "spyre_enabled": _group_by_operation(pure_xpass),
            "not_implemented": _group_by_operation(pure_xfail),
            "cpu_fallback": [{"operation": op} for op in sorted(cpu_fallback_ops)],
            "spyre_failed": _build_spyre_failed(xpass_list, xfail_list),
        },
        # Suite-level outcome (for ingest_model_ops.py aggregation)
        "suite_outcome": stats["outcome"],
        "suite_exit_code": stats["exit_code"],
        "suite_tests_total": stats["tests_total"],
        "suite_tests_passed": stats["tests_passed"],
        "suite_tests_failed": stats["tests_failed"],
        "suite_tests_skipped": stats["tests_skipped"],
        "suite_tests_error": stats["tests_error"],
        "suite_tests_xfail": stats["tests_xfail"],
        "suite_tests_xpass": stats["tests_xpass"],
        "suite_duration_s": stats["duration_s"],
        "triggered_at": _parse_ts(first_ts),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse model-ops GHA job logs → cleaned log + structured JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-dir",
        metavar="DIR",
        required=True,
        help="Directory containing raw GHA job log files (*.txt / *.log)",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="GHA run ID, e.g. 27674677047",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--log-out",
        metavar="FILE",
        default="model_ops_log.txt",
        help="Output cleaned plain-text log (default: model_ops_log.txt)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact (non-indented) JSON",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"[error] Not a directory: {log_dir}", file=sys.stderr)
        sys.exit(1)

    file_triples = _pick_log_files(log_dir)
    if not file_triples:
        print(f"[warn] No model-ops log files found in {log_dir}", file=sys.stderr)
        Path(args.out).write_text("[]")
        Path(args.log_out).write_text("")
        sys.exit(0)

    print(f"[info] Found {len(file_triples)} suite log file(s)", file=sys.stderr)

    all_records: list[dict] = []
    log_sections: list[str] = []

    for fpath, suite_name, model_name in file_triples:
        text = fpath.read_text(errors="replace")
        rec = parse_log_file(
            text, run_id=args.run_id, suite_name=suite_name, model_name=model_name
        )
        all_records.append(rec)

        # Summary for log file
        sep = "=" * 72
        xp = rec["summary"]["spyre_enabled_count"]
        xf = rec["summary"]["not_implemented_count"]
        fb = rec["summary"]["cpu_fallback_count"]
        sf = rec["summary"]["spyre_failed_count"]
        log_sections.append(
            f"{sep}\n"
            f"SUITE   : {suite_name}\n"
            f"MODEL   : {model_name}\n"
            f"FILE    : {fpath.name}\n"
            f"OUTCOME : {rec['suite_outcome']}\n"
            f"XPASS(spyre_enabled)={xp}  "
            f"XFAIL(not_implemented)={xf}  "
            f"fallback={fb}  "
            f"mixed(spyre_failed)={sf}\n"
            f"{sep}\n"
            f"{text.strip()}\n"
        )

        print(
            f"[info]  {fpath.name}  suite={suite_name!r}  "
            f"outcome={rec['suite_outcome']}  "
            f"xpass={xp}  xfail={xf}  fallback={fb}  mixed={sf}",
            file=sys.stderr,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    total = len(all_records)
    n_passed = sum(1 for r in all_records if r["suite_outcome"] == "passed")
    n_failed = sum(1 for r in all_records if r["suite_outcome"] == "failed")
    n_error = sum(1 for r in all_records if r["suite_outcome"] == "error")

    print("\n[info] ── Summary ──────────────────────────────────", file=sys.stderr)
    print(f"[info]  Total suites : {total}", file=sys.stderr)
    print(f"[info]  Passed       : {n_passed}", file=sys.stderr)
    print(f"[info]  Failed       : {n_failed}", file=sys.stderr)
    print(f"[info]  Error        : {n_error}", file=sys.stderr)

    # ── Write outputs ─────────────────────────────────────────────────────
    json_text = json.dumps(all_records, indent=None if args.compact else 2)
    Path(args.out).write_text(json_text)
    print(f"[info]  JSON written to : {args.out}", file=sys.stderr)

    Path(args.log_out).write_text("\n".join(log_sections))
    print(f"[info]  Log  written to : {args.log_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
