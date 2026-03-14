#!/usr/bin/env python3
"""
CyberGym result analysis.

Reads batch output produced by run_batch.py and prints summary statistics,
per-project breakdowns, trajectory analysis, behavioral insights, and
multi-batch comparisons.

Usage:
    python3.13 analyze_results.py results/batch_20260101_120000
    python3.13 analyze_results.py results/batch_* --sessions --csv
    python3.13 analyze_results.py results/batch_no_memrl results/batch_memrl --top-n 15
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class TaskResult:
    task_id: str
    status: str  # completed | timeout | error
    poc_found: bool
    poc_size: int
    project_name: str
    step_count: int
    tokens_in: int
    tokens_out: int
    elapsed: float
    had_memory: bool
    error: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskResult:
        metrics = d.get("metrics") or {}
        tokens = metrics.get("tokens") or {}
        return cls(
            task_id=d.get("task_id", ""),
            status=d.get("status", "unknown"),
            poc_found=bool(d.get("poc_found", False)),
            poc_size=int(d.get("poc_size", 0)),
            project_name=d.get("project_name", "unknown"),
            step_count=int(metrics.get("step_count", 0)),
            tokens_in=int(tokens.get("input", 0)),
            tokens_out=int(tokens.get("output", 0)),
            elapsed=float(d.get("elapsed", 0.0)),
            had_memory=bool(d.get("had_memory", False)),
            error=d.get("error", ""),
        )


@dataclass
class BatchData:
    name: str
    path: Path
    tasks: list[TaskResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def memrl_enabled(self) -> bool:
        return bool(self.config.get("memrl_enabled", False))


@dataclass
class ProjectStats:
    project_name: str
    total: int = 0
    poc_found: int = 0
    completed: int = 0
    timeout: int = 0
    error: int = 0
    avg_steps: float = 0.0
    avg_elapsed: float = 0.0

    @property
    def poc_rate(self) -> float:
        return self.poc_found / self.total * 100 if self.total else 0.0


@dataclass
class ToolCallStats:
    tool_name: str
    count: int = 0
    avg_per_session: float = 0.0


# ── Loading ──────────────────────────────────────────────────────────────────


def load_batch(batch_dir: Path, load_sessions: bool = False) -> BatchData:
    """Load all data from a single batch directory."""
    batch = BatchData(name=batch_dir.name, path=batch_dir)

    # Config
    config_path = batch_dir / "config.json"
    if config_path.exists():
        try:
            batch.config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s: %s", config_path, exc)

    # Summary
    summary_path = batch_dir / "summary.json"
    if summary_path.exists():
        try:
            batch.summary = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s: %s", summary_path, exc)

    # Per-task results
    tasks_dir = batch_dir / "tasks"
    if tasks_dir.is_dir():
        for task_file in sorted(tasks_dir.glob("*.json")):
            try:
                raw = json.loads(task_file.read_text())
                batch.tasks.append(TaskResult.from_dict(raw))
            except (json.JSONDecodeError, OSError, KeyError) as exc:
                logger.warning("Skipping %s: %s", task_file.name, exc)
    else:
        logger.warning("No tasks/ directory in %s", batch_dir)

    # Sessions (optional — can be large)
    if load_sessions:
        sessions_dir = batch_dir / "sessions"
        if sessions_dir.is_dir():
            for sess_file in sorted(sessions_dir.glob("*.json")):
                try:
                    batch.sessions[sess_file.stem] = json.loads(sess_file.read_text())
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Skipping session %s: %s", sess_file.name, exc)
            logger.info(
                "Loaded %d session files from %s",
                len(batch.sessions),
                batch_dir.name,
            )

    logger.info(
        "Loaded batch %s: %d tasks, memrl=%s",
        batch.name,
        len(batch.tasks),
        batch.memrl_enabled,
    )
    return batch


def resolve_batch_dirs(patterns: list[str]) -> list[Path]:
    """Expand CLI arguments (which may be globs) into concrete directories."""
    dirs: list[Path] = []
    for pattern in patterns:
        expanded = sorted(glob.glob(pattern))
        if not expanded:
            p = Path(pattern)
            if p.is_dir():
                expanded = [pattern]
            else:
                logger.warning("No match for pattern: %s", pattern)
                continue
        for entry in expanded:
            d = Path(entry)
            if d.is_dir():
                dirs.append(d)
    return dirs


# ── Summary statistics ───────────────────────────────────────────────────────


def compute_summary(tasks: list[TaskResult]) -> dict[str, Any]:
    """Aggregate counts and rates from a list of task results."""
    n = len(tasks)
    if n == 0:
        return {"total": 0}

    completed = sum(1 for t in tasks if t.status == "completed")
    timeout = sum(1 for t in tasks if t.status == "timeout")
    error = sum(1 for t in tasks if t.status == "error")
    poc_found = sum(1 for t in tasks if t.poc_found)
    with_memory = sum(1 for t in tasks if t.had_memory)

    total_steps = sum(t.step_count for t in tasks)
    total_tok_in = sum(t.tokens_in for t in tasks)
    total_tok_out = sum(t.tokens_out for t in tasks)
    total_elapsed = sum(t.elapsed for t in tasks)

    return {
        "total": n,
        "completed": completed,
        "timeout": timeout,
        "error": error,
        "poc_found": poc_found,
        "poc_rate": poc_found / n * 100,
        "with_memory": with_memory,
        "avg_steps": total_steps / n,
        "avg_tokens_in": total_tok_in / n,
        "avg_tokens_out": total_tok_out / n,
        "avg_elapsed": total_elapsed / n,
        "total_tokens_in": total_tok_in,
        "total_tokens_out": total_tok_out,
    }


# ── Per-project breakdown ───────────────────────────────────────────────────


def compute_project_stats(tasks: list[TaskResult]) -> list[ProjectStats]:
    """Group tasks by project and compute per-project metrics."""
    groups: dict[str, list[TaskResult]] = defaultdict(list)
    for t in tasks:
        groups[t.project_name].append(t)

    stats: list[ProjectStats] = []
    for project, group in groups.items():
        n = len(group)
        ps = ProjectStats(
            project_name=project,
            total=n,
            poc_found=sum(1 for t in group if t.poc_found),
            completed=sum(1 for t in group if t.status == "completed"),
            timeout=sum(1 for t in group if t.status == "timeout"),
            error=sum(1 for t in group if t.status == "error"),
            avg_steps=sum(t.step_count for t in group) / n,
            avg_elapsed=sum(t.elapsed for t in group) / n,
        )
        stats.append(ps)

    # Sort by poc_rate ascending → hardest first
    stats.sort(key=lambda s: (s.poc_rate, s.project_name))
    return stats


# ── Session / trajectory analysis ────────────────────────────────────────────


KNOWN_TOOLS = ("bash", "read", "write", "edit", "glob", "grep")


def analyze_sessions(
    sessions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Extract tool-call counts and trajectory statistics from session data."""
    tool_counts: dict[str, int] = defaultdict(int)
    total_steps = 0
    total_tool_calls = 0
    session_count = 0

    for _name, session in sessions.items():
        messages = _extract_messages(session)
        if not messages:
            continue
        session_count += 1

        for msg in messages:
            role = msg.get("role", "")
            if role == "assistant":
                calls = msg.get("tool_calls") or msg.get("function_calls") or []
                if isinstance(calls, list):
                    for call in calls:
                        tool_name = _extract_tool_name(call)
                        if tool_name:
                            tool_counts[tool_name] += 1
                            total_tool_calls += 1
                total_steps += 1

    tool_stats: list[ToolCallStats] = []
    for name in sorted(tool_counts, key=lambda k: tool_counts[k], reverse=True):
        tool_stats.append(
            ToolCallStats(
                tool_name=name,
                count=tool_counts[name],
                avg_per_session=(
                    tool_counts[name] / session_count if session_count else 0
                ),
            )
        )

    return {
        "session_count": session_count,
        "total_tool_calls": total_tool_calls,
        "total_steps": total_steps,
        "avg_steps_per_session": total_steps / session_count if session_count else 0,
        "avg_tool_calls_per_session": (
            total_tool_calls / session_count if session_count else 0
        ),
        "tool_stats": tool_stats,
    }


def _extract_messages(session: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    """Flexibly pull the messages list out of a session structure."""
    if isinstance(session, list):
        return session
    if isinstance(session, dict):
        for key in ("messages", "trajectory", "history", "conversation"):
            if key in session and isinstance(session[key], list):
                return session[key]
        # Might be the messages directly at top level
        if "role" in session:
            return [session]
    return []


def _extract_tool_name(call: Any) -> str:
    """Pull tool name from various call formats."""
    if isinstance(call, dict):
        name = call.get("name") or call.get("function", {}).get("name", "")
        if isinstance(name, str):
            return name
    return ""


# ── Behavioral analysis ─────────────────────────────────────────────────────


def compute_behavioral(tasks: list[TaskResult]) -> dict[str, Any]:
    """Compute behavioral metrics: poc creation rate, idle timeouts, etc."""
    n = len(tasks)
    if n == 0:
        return {}

    created_poc = sum(1 for t in tasks if t.poc_found or t.poc_size > 0)
    timeout_no_write = sum(
        1
        for t in tasks
        if t.status == "timeout" and not t.poc_found and t.poc_size == 0
    )
    timeout_total = sum(1 for t in tasks if t.status == "timeout")
    error_total = sum(1 for t in tasks if t.status == "error")
    poc_but_wrong = sum(1 for t in tasks if t.poc_size > 0 and not t.poc_found)

    return {
        "created_poc_file": created_poc,
        "created_poc_rate": created_poc / n * 100,
        "timeout_no_write": timeout_no_write,
        "timeout_no_write_rate": (
            timeout_no_write / timeout_total * 100 if timeout_total else 0
        ),
        "timeout_total": timeout_total,
        "error_total": error_total,
        "poc_but_wrong": poc_but_wrong,
        "poc_but_wrong_rate": poc_but_wrong / n * 100,
    }


# ── Table formatting ─────────────────────────────────────────────────────────


def format_table(
    headers: list[str],
    rows: list[list[str]],
    alignments: list[str] | None = None,
) -> str:
    """Render a simple ASCII table.

    *alignments* is a list of "<", ">", or "^" per column.
    """
    if not rows:
        return "(no data)\n"

    n_cols = len(headers)
    if alignments is None:
        alignments = ["<"] * n_cols

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row[:n_cols]):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: list[str], widths: list[int], aligns: list[str]) -> str:
        parts: list[str] = []
        for cell, w, a in zip(cells, widths, aligns):
            if a == ">":
                parts.append(cell.rjust(w))
            elif a == "^":
                parts.append(cell.center(w))
            else:
                parts.append(cell.ljust(w))
        return "  ".join(parts)

    sep = "  ".join("-" * w for w in col_widths)
    lines = [
        fmt_row(headers, col_widths, alignments),
        sep,
    ]
    for row in rows:
        padded = row + [""] * (n_cols - len(row))
        lines.append(fmt_row(padded, col_widths, alignments))
    return "\n".join(lines) + "\n"


def pct(num: float, den: float) -> str:
    """Format a percentage with one decimal place."""
    if den == 0:
        return "—"
    return f"{num / den * 100:.1f}%"


def fmt_int(n: int | float) -> str:
    return f"{int(n):,}"


def fmt_float(n: float, decimals: int = 1) -> str:
    return f"{n:.{decimals}f}"


# ── Printers ─────────────────────────────────────────────────────────────────


def print_summary_table(batch: BatchData) -> None:
    s = compute_summary(batch.tasks)
    if s["total"] == 0:
        print(f"\n[{batch.name}]  (no tasks)")
        return

    print(f"\n{'=' * 64}")
    print(f"  Batch: {batch.name}")
    print(f"  MEMRL: {'enabled' if batch.memrl_enabled else 'disabled'}")
    print(f"{'=' * 64}")

    headers = ["Metric", "Value"]
    rows = [
        ["Total tasks", fmt_int(s["total"])],
        ["Completed", f"{fmt_int(s['completed'])} ({pct(s['completed'], s['total'])})"],
        ["Timeout", f"{fmt_int(s['timeout'])} ({pct(s['timeout'], s['total'])})"],
        ["Error", f"{fmt_int(s['error'])} ({pct(s['error'], s['total'])})"],
        ["PoC found", f"{fmt_int(s['poc_found'])} ({pct(s['poc_found'], s['total'])})"],
        ["With memory", fmt_int(s["with_memory"])],
        ["Avg steps", fmt_float(s["avg_steps"])],
        ["Avg tokens (in)", fmt_int(s["avg_tokens_in"])],
        ["Avg tokens (out)", fmt_int(s["avg_tokens_out"])],
        ["Avg elapsed (s)", fmt_float(s["avg_elapsed"])],
        ["Total tokens in", fmt_int(s["total_tokens_in"])],
        ["Total tokens out", fmt_int(s["total_tokens_out"])],
    ]
    print(format_table(headers, rows, ["<", ">"]))


def print_project_table(
    batch: BatchData,
    top_n: int,
    label: str = "",
) -> list[ProjectStats]:
    stats = compute_project_stats(batch.tasks)
    if not stats:
        return []

    title = f"Per-Project Stats ({label or batch.name})"
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")

    headers = [
        "Project",
        "Total",
        "PoC",
        "Rate",
        "Done",
        "Tmo",
        "Err",
        "Avg Steps",
        "Avg Time",
    ]
    aligns = ["<", ">", ">", ">", ">", ">", ">", ">", ">"]

    # Hardest
    hardest = stats[:top_n]
    print(f"\n  Top {min(top_n, len(hardest))} hardest (lowest PoC rate):")
    rows = [
        [
            ps.project_name[:30],
            str(ps.total),
            str(ps.poc_found),
            f"{ps.poc_rate:.1f}%",
            str(ps.completed),
            str(ps.timeout),
            str(ps.error),
            fmt_float(ps.avg_steps),
            f"{ps.avg_elapsed:.0f}s",
        ]
        for ps in hardest
    ]
    print(format_table(headers, rows, aligns))

    # Easiest
    easiest = list(reversed(stats[-top_n:]))
    if easiest and easiest != hardest:
        print(f"  Top {min(top_n, len(easiest))} easiest (highest PoC rate):")
        rows = [
            [
                ps.project_name[:30],
                str(ps.total),
                str(ps.poc_found),
                f"{ps.poc_rate:.1f}%",
                str(ps.completed),
                str(ps.timeout),
                str(ps.error),
                fmt_float(ps.avg_steps),
                f"{ps.avg_elapsed:.0f}s",
            ]
            for ps in easiest
        ]
        print(format_table(headers, rows, aligns))

    return stats


def print_session_analysis(batch: BatchData) -> dict[str, Any] | None:
    if not batch.sessions:
        return None

    result = analyze_sessions(batch.sessions)
    if result["session_count"] == 0:
        print(f"\n  Session analysis: no parseable sessions in {batch.name}")
        return result

    print(f"\n{'─' * 64}")
    print(f"  Trajectory Analysis ({batch.name})")
    print(f"{'─' * 64}")
    print(f"  Sessions analyzed:        {result['session_count']}")
    print(f"  Avg steps / session:      {result['avg_steps_per_session']:.1f}")
    print(f"  Avg tool calls / session: {result['avg_tool_calls_per_session']:.1f}")
    print(f"  Total tool calls:         {fmt_int(result['total_tool_calls'])}")

    if result["tool_stats"]:
        print()
        headers = ["Tool", "Count", "Avg/Session"]
        rows = [
            [
                ts.tool_name,
                fmt_int(ts.count),
                fmt_float(ts.avg_per_session),
            ]
            for ts in result["tool_stats"][:20]
        ]
        print(format_table(headers, rows, ["<", ">", ">"]))

    return result


def print_behavioral(batch: BatchData) -> dict[str, Any] | None:
    bh = compute_behavioral(batch.tasks)
    if not bh:
        return None

    print(f"\n{'─' * 64}")
    print(f"  Behavioral Analysis ({batch.name})")
    print(f"{'─' * 64}")

    headers = ["Metric", "Count", "Rate"]
    rows = [
        [
            "Created PoC file",
            fmt_int(bh["created_poc_file"]),
            f"{bh['created_poc_rate']:.1f}%",
        ],
        [
            "PoC exists but wrong",
            fmt_int(bh["poc_but_wrong"]),
            f"{bh['poc_but_wrong_rate']:.1f}%",
        ],
        [
            "Timeout w/o writing",
            fmt_int(bh["timeout_no_write"]),
            f"{bh['timeout_no_write_rate']:.1f}% of timeouts",
        ],
        [
            "Total timeouts",
            fmt_int(bh["timeout_total"]),
            "",
        ],
        [
            "Total errors",
            fmt_int(bh["error_total"]),
            "",
        ],
    ]
    print(format_table(headers, rows, ["<", ">", ">"]))
    return bh


# ── Multi-batch comparison ───────────────────────────────────────────────────


def print_comparison(batches: list[BatchData]) -> None:
    """Side-by-side comparison of multiple batches."""
    if len(batches) < 2:
        return

    print(f"\n{'=' * 80}")
    print("  Multi-Batch Comparison")
    print(f"{'=' * 80}")

    summaries = [(b.name, compute_summary(b.tasks), b.memrl_enabled) for b in batches]

    headers = ["Metric"] + [name for name, _, _ in summaries]
    aligns = ["<"] + [">"] * len(summaries)

    metric_keys: list[tuple[str, str]] = [
        ("total", "Total tasks"),
        ("completed", "Completed"),
        ("timeout", "Timeout"),
        ("error", "Error"),
        ("poc_found", "PoC found"),
        ("poc_rate", "PoC rate (%)"),
        ("with_memory", "With memory"),
        ("avg_steps", "Avg steps"),
        ("avg_tokens_in", "Avg tokens in"),
        ("avg_tokens_out", "Avg tokens out"),
        ("avg_elapsed", "Avg elapsed (s)"),
    ]

    rows: list[list[str]] = []
    # MEMRL status row
    rows.append(["MEMRL"] + ["yes" if enabled else "no" for _, _, enabled in summaries])

    for key, label in metric_keys:
        row = [label]
        for _, s, _ in summaries:
            val = s.get(key, 0)
            if isinstance(val, float):
                if key == "poc_rate":
                    row.append(f"{val:.1f}%")
                else:
                    row.append(fmt_float(val))
            else:
                row.append(fmt_int(val))
        rows.append(row)

    print(format_table(headers, rows, aligns))

    # Delta between first two batches
    if len(summaries) >= 2:
        _, s_a, _ = summaries[0]
        _, s_b, _ = summaries[1]
        if s_a.get("total", 0) > 0 and s_b.get("total", 0) > 0:
            delta_poc = (s_b.get("poc_rate", 0) or 0) - (s_a.get("poc_rate", 0) or 0)
            sign = "+" if delta_poc >= 0 else ""
            print(
                f"  PoC rate delta ({summaries[1][0]} vs {summaries[0][0]}): "
                f"{sign}{delta_poc:.1f} percentage points"
            )

    # Per-project comparison between first two batches
    if len(batches) >= 2:
        _print_project_comparison(batches[0], batches[1])


def _print_project_comparison(batch_a: BatchData, batch_b: BatchData) -> None:
    """Show per-project PoC rate diff between two batches."""
    stats_a = {ps.project_name: ps for ps in compute_project_stats(batch_a.tasks)}
    stats_b = {ps.project_name: ps for ps in compute_project_stats(batch_b.tasks)}

    all_projects = sorted(set(stats_a) | set(stats_b))
    if not all_projects:
        return

    print(f"\n{'─' * 80}")
    print(f"  Per-Project Comparison ({batch_a.name} vs {batch_b.name})")
    print(f"{'─' * 80}")

    headers = [
        "Project",
        f"Rate A ({batch_a.name[:15]})",
        f"Rate B ({batch_b.name[:15]})",
        "Delta",
    ]
    aligns = ["<", ">", ">", ">"]

    diffs: list[tuple[str, float, float, float]] = []
    for proj in all_projects:
        rate_a = stats_a[proj].poc_rate if proj in stats_a else 0.0
        rate_b = stats_b[proj].poc_rate if proj in stats_b else 0.0
        diffs.append((proj, rate_a, rate_b, rate_b - rate_a))

    # Sort by absolute delta descending — biggest movers first
    diffs.sort(key=lambda x: abs(x[3]), reverse=True)

    rows = [
        [
            proj[:30],
            f"{ra:.1f}%",
            f"{rb:.1f}%",
            f"{'+' if d >= 0 else ''}{d:.1f}pp",
        ]
        for proj, ra, rb, d in diffs[:20]
    ]
    print(format_table(headers, rows, aligns))


# ── CSV export ───────────────────────────────────────────────────────────────


def write_csv_summary(batch: BatchData, output_dir: Path) -> None:
    """Write per-task CSV and project-summary CSV for a batch."""
    # Per-task
    task_csv = output_dir / f"{batch.name}_tasks.csv"
    with open(task_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task_id",
                "status",
                "poc_found",
                "poc_size",
                "project_name",
                "step_count",
                "tokens_in",
                "tokens_out",
                "elapsed",
                "had_memory",
                "error",
            ]
        )
        for t in batch.tasks:
            writer.writerow(
                [
                    t.task_id,
                    t.status,
                    t.poc_found,
                    t.poc_size,
                    t.project_name,
                    t.step_count,
                    t.tokens_in,
                    t.tokens_out,
                    t.elapsed,
                    t.had_memory,
                    t.error,
                ]
            )
    logger.info("Wrote %s", task_csv)

    # Per-project
    project_csv = output_dir / f"{batch.name}_projects.csv"
    stats = compute_project_stats(batch.tasks)
    with open(project_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "project_name",
                "total",
                "poc_found",
                "poc_rate",
                "completed",
                "timeout",
                "error",
                "avg_steps",
                "avg_elapsed",
            ]
        )
        for ps in stats:
            writer.writerow(
                [
                    ps.project_name,
                    ps.total,
                    ps.poc_found,
                    f"{ps.poc_rate:.2f}",
                    ps.completed,
                    ps.timeout,
                    ps.error,
                    f"{ps.avg_steps:.1f}",
                    f"{ps.avg_elapsed:.1f}",
                ]
            )
    logger.info("Wrote %s", project_csv)


def write_csv_comparison(batches: list[BatchData], output_dir: Path) -> None:
    """Write a comparison CSV when multiple batches are loaded."""
    if len(batches) < 2:
        return

    cmp_csv = output_dir / "comparison.csv"
    summaries = [(b.name, compute_summary(b.tasks), b.memrl_enabled) for b in batches]

    with open(cmp_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + [name for name, _, _ in summaries])
        writer.writerow(["memrl_enabled"] + [str(en) for _, _, en in summaries])
        for key in (
            "total",
            "completed",
            "timeout",
            "error",
            "poc_found",
            "poc_rate",
            "with_memory",
            "avg_steps",
            "avg_tokens_in",
            "avg_tokens_out",
            "avg_elapsed",
        ):
            row = [key]
            for _, s, _ in summaries:
                val = s.get(key, 0)
                if isinstance(val, float):
                    row.append(f"{val:.2f}")
                else:
                    row.append(str(val))
            writer.writerow(row)
    logger.info("Wrote %s", cmp_csv)


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze CyberGym batch results from run_batch.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s results/batch_20260101_120000\n"
            "  %(prog)s results/batch_* --sessions --csv\n"
            "  %(prog)s results/batch_no_memrl results/batch_memrl --top-n 15\n"
        ),
    )
    p.add_argument(
        "batch_dirs",
        nargs="+",
        help="One or more batch directories (supports glob patterns)",
    )
    p.add_argument(
        "--csv",
        action="store_true",
        help="Write CSV files alongside terminal output",
    )
    p.add_argument(
        "--sessions",
        action="store_true",
        help="Include trajectory analysis (reads session files, slower)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of hardest/easiest projects to show (default: 10)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dirs = resolve_batch_dirs(args.batch_dirs)
    if not dirs:
        logger.error("No valid batch directories found.")
        sys.exit(1)

    logger.info("Loading %d batch(es): %s", len(dirs), [d.name for d in dirs])

    batches: list[BatchData] = []
    for d in dirs:
        batches.append(load_batch(d, load_sessions=args.sessions))

    # ── Per-batch reports ────────────────────────────────────────────────
    for batch in batches:
        print_summary_table(batch)
        print_project_table(batch, top_n=args.top_n)
        print_behavioral(batch)
        if args.sessions:
            print_session_analysis(batch)

    # ── Multi-batch comparison ───────────────────────────────────────────
    if len(batches) >= 2:
        print_comparison(batches)

    # ── CSV export ───────────────────────────────────────────────────────
    if args.csv:
        csv_dir = Path(".")
        for batch in batches:
            write_csv_summary(batch, csv_dir)
        if len(batches) >= 2:
            write_csv_comparison(batches, csv_dir)

    print()


if __name__ == "__main__":
    main()
