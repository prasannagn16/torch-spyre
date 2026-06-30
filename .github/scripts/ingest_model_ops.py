#!/usr/bin/env python3
"""
Reads the JSON produced by parse_model_ops_logs.py and batch-inserts the
rows into two ClickHouse tables:

  model_ops_suites  – one row per model suite per GHA run
                      (suite outcome + counts)

  model_ops_variants – one row per individual test variant
                       (operation, shapes, dtypes, XPASS/XFAIL/FALLBACK,
                        matching exactly the JSON schema shown in the dashboard)

Both tables are created automatically on first run (CREATE TABLE IF NOT EXISTS).

Usage (called by the GHA workflow):
    python3 ingest_model_ops.py \\
        --json-file model_ops_model-ops-tests_27674677047.json \\
        --workflow  "model-ops-tests" \\
        --branch    "main" \\
        --sha       "abc123..." \\
        --run-id    "27674677047"
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import clickhouse_connect

# ---------------------------------------------------------------------------
# ClickHouse DDL
# ---------------------------------------------------------------------------

_CREATE_SUITES_SQL = """
CREATE TABLE IF NOT EXISTS model_ops_suites
(
    -- Identity / provenance
    gha_run_id      UInt64,
    run_id          String,
    workflow        LowCardinality(String) DEFAULT '',
    branch          LowCardinality(String) DEFAULT '',
    commit_sha      String DEFAULT '',

    -- Suite / config
    suite_name      String,
    model_name      LowCardinality(String) DEFAULT '',
    yaml_file       String DEFAULT '',

    -- Counts
    total_tests          UInt32 DEFAULT 0,
    spyre_enabled_count  UInt32 DEFAULT 0,
    not_implemented_count UInt32 DEFAULT 0,
    cpu_fallback_count   UInt32 DEFAULT 0,
    spyre_failed_count   UInt32 DEFAULT 0,

    -- Suite-level pytest stats
    suite_outcome   LowCardinality(String) DEFAULT 'unknown',
    suite_exit_code Nullable(Int32),
    tests_total     UInt32 DEFAULT 0,
    tests_passed    UInt32 DEFAULT 0,
    tests_failed    UInt32 DEFAULT 0,
    tests_skipped   UInt32 DEFAULT 0,
    tests_error     UInt32 DEFAULT 0,
    tests_xfail     UInt32 DEFAULT 0,
    tests_xpass     UInt32 DEFAULT 0,
    duration_s      Float32 DEFAULT 0,

    -- Timestamps
    triggered_at    DateTime64(3, 'UTC'),
    ingested_at     DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(ingested_at)
ORDER BY (gha_run_id, suite_name)
PARTITION BY toYYYYMM(triggered_at)
SETTINGS index_granularity = 8192
"""

_CREATE_VARIANTS_SQL = """
CREATE TABLE IF NOT EXISTS model_ops_variants
(
    -- Provenance (joins back to model_ops_suites)
    gha_run_id      UInt64,
    run_id          String,
    workflow        LowCardinality(String) DEFAULT '',
    branch          LowCardinality(String) DEFAULT '',
    commit_sha      String DEFAULT '',

    -- Suite / config
    suite_name      String,
    model_name      LowCardinality(String) DEFAULT '',
    yaml_file       String DEFAULT '',

    -- Variant identity
    operation       LowCardinality(String),
    classification  LowCardinality(String),   -- spyre_enabled | not_implemented | cpu_fallback
    test_name       String,
    status          LowCardinality(String),   -- XPASS | XFAIL | FALLBACK

    -- Tensor info (stored as JSON arrays serialised as strings)
    input_shapes    String DEFAULT '[]',      -- JSON array of shape strings
    input_strides   String DEFAULT '[]',
    input_dtypes    String DEFAULT '[]',
    arg_values      String DEFAULT '[]',
    target_shape    String DEFAULT '',

    -- Timestamps
    triggered_at    DateTime64(3, 'UTC'),
    ingested_at     DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(ingested_at)
ORDER BY (gha_run_id, suite_name, test_name)
PARTITION BY toYYYYMM(triggered_at)
SETTINGS index_granularity = 8192
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_client():
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 443)),
        user=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ["CLICKHOUSE_PASS"],
        database=os.environ.get("CLICKHOUSE_DB", "spyre"),
        secure=True,
    )


def _parse_ts(ts_str: str) -> datetime:
    """ISO-8601 string → naive UTC datetime."""
    if not ts_str:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None)
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc).replace(tzinfo=None)


def _str(val, default: str = "") -> str:
    return str(val).strip() if val is not None else default


def _int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _jstr(val) -> str:
    """Serialise a list → JSON string for ClickHouse String column."""
    if not val:
        return "[]"
    if isinstance(val, str):
        return val  # already serialised
    try:
        return json.dumps(val, ensure_ascii=False)
    except (TypeError, ValueError):
        return "[]"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def suite_already_ingested(client, gha_run_id: int, suite_name: str) -> bool:
    result = client.query(
        "SELECT count() FROM model_ops_suites "
        "WHERE gha_run_id = {r:UInt64} AND suite_name = {s:String}",
        parameters={"r": gha_run_id, "s": suite_name},
    )
    return result.result_rows[0][0] > 0


def variants_already_ingested(client, gha_run_id: int, suite_name: str) -> bool:
    result = client.query(
        "SELECT count() FROM model_ops_variants "
        "WHERE gha_run_id = {r:UInt64} AND suite_name = {s:String}",
        parameters={"r": gha_run_id, "s": suite_name},
    )
    return result.result_rows[0][0] > 0


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

SUITE_COLS = [
    "gha_run_id",
    "run_id",
    "workflow",
    "branch",
    "commit_sha",
    "suite_name",
    "model_name",
    "yaml_file",
    "total_tests",
    "spyre_enabled_count",
    "not_implemented_count",
    "cpu_fallback_count",
    "spyre_failed_count",
    "suite_outcome",
    "suite_exit_code",
    "tests_total",
    "tests_passed",
    "tests_failed",
    "tests_skipped",
    "tests_error",
    "tests_xfail",
    "tests_xpass",
    "duration_s",
    "triggered_at",
    "ingested_at",
]

VARIANT_COLS = [
    "gha_run_id",
    "run_id",
    "workflow",
    "branch",
    "commit_sha",
    "suite_name",
    "model_name",
    "yaml_file",
    "operation",
    "classification",
    "test_name",
    "status",
    "input_shapes",
    "input_strides",
    "input_dtypes",
    "arg_values",
    "target_shape",
    "triggered_at",
    "ingested_at",
]


def build_suite_row(rec: dict, args, gha_run_id: int, now: datetime) -> list:
    summary = rec.get("summary", {})
    return [
        gha_run_id,
        _str(rec.get("run_id") or args.run_id),
        _str(args.workflow),
        _str(args.branch),
        _str(args.sha)[:40].ljust(40)[:40],
        _str(rec.get("suite_name")),
        _str(rec.get("model_name")),
        _str(rec.get("yaml_file")),
        _int(summary.get("total_tests")),
        _int(summary.get("spyre_enabled_count")),
        _int(summary.get("not_implemented_count")),
        _int(summary.get("cpu_fallback_count")),
        _int(summary.get("spyre_failed_count")),
        _str(rec.get("suite_outcome"), "unknown"),
        rec.get("suite_exit_code"),  # Nullable(Int32) — keep None
        _int(rec.get("suite_tests_total")),
        _int(rec.get("suite_tests_passed")),
        _int(rec.get("suite_tests_failed")),
        _int(rec.get("suite_tests_skipped")),
        _int(rec.get("suite_tests_error")),
        _int(rec.get("suite_tests_xfail")),
        _int(rec.get("suite_tests_xpass")),
        _float(rec.get("suite_duration_s")),
        _parse_ts(rec.get("triggered_at")),
        now,
    ]


def _build_variant_rows(
    rec: dict,
    args,
    gha_run_id: int,
    now: datetime,
) -> list[list]:
    """
    Flatten all per-variant records in a suite record into a list of rows
    for model_ops_variants.

    Covers: spyre_enabled, not_implemented variants (from operations.spyre_enabled
    / not_implemented / spyre_failed), and cpu_fallback entries.
    """
    suite_name = _str(rec.get("suite_name"))
    model_name = _str(rec.get("model_name"))
    yaml_file = _str(rec.get("yaml_file"))
    run_id = _str(rec.get("run_id") or args.run_id)
    triggered_at = _parse_ts(rec.get("triggered_at"))

    rows: list[list] = []
    ops = rec.get("operations", {})

    def _base():
        return [
            gha_run_id,
            run_id,
            _str(args.workflow),
            _str(args.branch),
            _str(args.sha)[:40].ljust(40)[:40],
            suite_name,
            model_name,
            yaml_file,
        ]

    def _variant_row(v: dict, classification: str, status: str) -> list:
        return _base() + [
            _str(v.get("operation")),
            classification,
            _str(v.get("test_name")),
            status,
            _jstr(v.get("input_shapes", [])),
            _jstr(v.get("input_strides", [])),
            _jstr(v.get("input_dtypes", [])),
            _jstr(v.get("arg_values", [])),
            _str(v.get("target_shape", "")),
            triggered_at,
            now,
        ]

    # ── spyre_enabled groups ────────────────────────────────────────────────
    for group in ops.get("spyre_enabled", []):
        for v in group.get("variants", []):
            rows.append(_variant_row(v, "spyre_enabled", "XPASS"))

    # ── not_implemented groups ──────────────────────────────────────────────
    for group in ops.get("not_implemented", []):
        for v in group.get("variants", []):
            rows.append(_variant_row(v, "not_implemented", "XFAIL"))

    # ── spyre_failed groups (mixed: some XPASS, some XFAIL) ────────────────
    for group in ops.get("spyre_failed", []):
        for v in group.get("xpass_variants", []):
            rows.append(_variant_row(v, "spyre_enabled", "XPASS"))
        for v in group.get("xfail_variants", []):
            rows.append(_variant_row(v, "not_implemented", "XFAIL"))

    # ── cpu_fallback (no test_name; just op name) ───────────────────────────
    for entry in ops.get("cpu_fallback", []):
        op = _str(entry.get("operation"))
        if not op:
            continue
        rows.append(
            _base()
            + [
                op,
                "cpu_fallback",
                "",  # no test_name for fallback entries
                "FALLBACK",
                "[]",
                "[]",
                "[]",
                "[]",
                "",
                triggered_at,
                now,
            ]
        )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest model_ops JSON → ClickHouse (model_ops_suites + model_ops_variants)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--json-file", required=True, help="Path to JSON from parse_model_ops_logs.py"
    )
    parser.add_argument(
        "--workflow", default="model-ops-tests", help="GHA workflow name"
    )
    parser.add_argument("--branch", default="", help="Git branch name")
    parser.add_argument("--sha", default="", help="Git commit SHA")
    parser.add_argument("--run-id", default="", help="GHA run ID (numeric string)")
    args = parser.parse_args()

    # ── Load JSON ────────────────────────────────────────────────────────────
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"[error] File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as fh:
        raw = json.load(fh)

    # Accept two formats:
    #   (A) flat array  – produced by parse_model_ops_logs.py --ingest-json
    #   (B) dashboard envelope – {"total_models": N, "models": [...]}
    #       produced by parse_model_ops_logs.py --out  (legacy / CI path)
    if isinstance(raw, dict) and "models" in raw:
        # Dashboard envelope: models lack suite_name / run_id / suite_outcome.
        # Reconstruct minimal ingest-compatible records from dashboard fields.
        records = []
        for m in raw.get("models", []):
            rec = dict(m)
            # Derive suite_name from model_name if absent
            if not rec.get("suite_name"):
                rec["suite_name"] = rec.get("model_name", "")
            # Provide empty ingest-only fields if absent
            rec.setdefault("run_id", args.run_id)
            rec.setdefault("suite_outcome", "unknown")
            rec.setdefault("suite_exit_code", None)
            rec.setdefault(
                "suite_tests_total", rec.get("summary", {}).get("total_tests", 0)
            )
            rec.setdefault("suite_tests_passed", 0)
            rec.setdefault("suite_tests_failed", 0)
            rec.setdefault("suite_tests_skipped", 0)
            rec.setdefault("suite_tests_error", 0)
            rec.setdefault("suite_tests_xfail", 0)
            rec.setdefault(
                "suite_tests_xpass",
                rec.get("summary", {}).get("spyre_enabled_count", 0),
            )
            rec.setdefault("suite_duration_s", 0.0)
            rec.setdefault("triggered_at", "")
            rec.setdefault("ingested_at", "")
            records.append(rec)
        print(f"[info] Dashboard-envelope format detected: {len(records)} model(s)")
    elif isinstance(raw, list):
        records = raw
        print(f"[info] Flat-array format detected: {len(records)} suite record(s)")
    else:
        print(
            "[error] Unrecognised JSON format — expected list or {models:[...]} dict",
            file=sys.stderr,
        )
        sys.exit(1)

    if not records:
        print("[info] JSON file contains no records — nothing to ingest.")
        sys.exit(0)

    # Filter out records without a suite_name
    records = [
        r
        for r in records
        if r.get("suite_name", "").strip() and not r["suite_name"].startswith(".")
    ]
    if not records:
        print("[info] No valid records after filtering — nothing to ingest.")
        sys.exit(0)

    print(f"[info] Loaded {len(records)} suite record(s) from {json_path.name}")

    # ── Connect ──────────────────────────────────────────────────────────────
    print(
        f"[info] Connecting to ClickHouse at "
        f"{os.environ['CLICKHOUSE_HOST']}:{os.environ.get('CLICKHOUSE_PORT', 443)} ..."
    )
    client = get_client()
    client.command("SELECT 1")
    print("[info] Connected.\n")

    # ── Ensure tables exist ───────────────────────────────────────────────────
    print("[info] Ensuring tables exist ...")
    client.command(_CREATE_SUITES_SQL)
    client.command(_CREATE_VARIANTS_SQL)
    print("[info] Tables ready.\n")

    gha_run_id = _int(args.run_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    suites_inserted = 0
    suites_skipped = 0
    variants_inserted = 0
    variants_skipped = 0

    for rec in records:
        suite_name = _str(rec.get("suite_name"))
        model_name = _str(rec.get("model_name"))

        # ── Suite row ───────────────────────────────────────────────────────
        if suite_already_ingested(client, gha_run_id, suite_name):
            print(f"  [skip-suite]    gha_run_id={gha_run_id} suite={suite_name!r}")
            suites_skipped += 1
        else:
            try:
                suite_row = build_suite_row(rec, args, gha_run_id, now)
                client.insert("model_ops_suites", [suite_row], column_names=SUITE_COLS)
                suites_inserted += 1
                print(
                    f"  [suite ok]   {suite_name!r}  model={model_name}  "
                    f"outcome={rec.get('suite_outcome')}  "
                    f"xpass={rec.get('summary', {}).get('spyre_enabled_count', 0)}  "
                    f"xfail={rec.get('summary', {}).get('not_implemented_count', 0)}"
                )
            except Exception as exc:
                print(f"  [suite err]  {suite_name!r}: {exc}", file=sys.stderr)
                suites_skipped += 1

        # ── Variant rows ────────────────────────────────────────────────────
        if variants_already_ingested(client, gha_run_id, suite_name):
            print(f"  [skip-variants] gha_run_id={gha_run_id} suite={suite_name!r}")
            variants_skipped += 1
        else:
            try:
                variant_rows = _build_variant_rows(rec, args, gha_run_id, now)
                if variant_rows:
                    client.insert(
                        "model_ops_variants",
                        variant_rows,
                        column_names=VARIANT_COLS,
                    )
                    variants_inserted += len(variant_rows)
                    print(f"  [variants ok] {suite_name!r}: {len(variant_rows)} rows")
                else:
                    print(f"  [variants 0]  {suite_name!r}: no variants found in JSON")
            except Exception as exc:
                print(f"  [variants err] {suite_name!r}: {exc}", file=sys.stderr)
                variants_skipped += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n[info] Ingest complete")
    print(
        f"[info]   model_ops_suites   inserted={suites_inserted}   skipped={suites_skipped}"
    )
    print(
        f"[info]   model_ops_variants inserted={variants_inserted}  skipped={variants_skipped}"
    )
    print(f"[info]   gha_run_id  : {gha_run_id}")
    print(f"[info]   workflow    : {args.workflow}")
    print(f"[info]   branch      : {args.branch}")
    print(f"[info]   sha         : {args.sha[:12]}")


if __name__ == "__main__":
    main()
