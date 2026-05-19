#!/usr/bin/env python3
"""
gap_runner.py — run a Cayley-DSRG search via a GAP subprocess with live progress.

Usage:
    python gap_runner.py n k t lambda mu [--timeout SECONDS] [--log FILE]
"""

import json
import logging
import subprocess
import sys
import tempfile
import threading
import time
from math import comb
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Progress line renderer
# ─────────────────────────────────────────────────────────────────────────────

class ProgressLine:
    """Renders a single overwriting status line on stdout.

    Log messages are printed above it cleanly: the progress line is erased
    before each log record is emitted, then redrawn afterwards.
    """

    CLEAR = "\r\033[K"

    def __init__(self) -> None:
        self._current = ""

    def update(self, msg: str) -> None:
        self._current = msg
        print(f"{self.CLEAR}{msg}", end="", flush=True)

    def clear(self) -> None:
        print(self.CLEAR, end="", flush=True)

    def redraw(self) -> None:
        if self._current:
            print(f"{self.CLEAR}{self._current}", end="", flush=True)

    def finish(self) -> None:
        """Move to a new line (call when done or before a permanent message)."""
        if self._current:
            print()
            self._current = ""


class _ProgressAwareStreamHandler(logging.StreamHandler):
    """Stream handler that erases the progress line before each record
    and redraws it afterwards so log output never clobbers the status."""

    def __init__(self, progress: ProgressLine, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._progress = progress

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        self._progress.clear()
        super().emit(record)
        self._progress.redraw()


# ─────────────────────────────────────────────────────────────────────────────
# Logger setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(logfile: Path, progress: ProgressLine) -> logging.Logger:
    logger = logging.getLogger("cayley_search")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = _ProgressAwareStreamHandler(progress)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# GAP script template
# ─────────────────────────────────────────────────────────────────────────────
#
# Structured tokens written to stdout:
#   GROUPS  <total>
#   GROUP   <idx> <total> <StructureDescription>
#   SET     <current> <total>
#   FOUND
#   ADJ_START
#   <row as comma-separated ints>  (one per line, n lines)
#   ADJ_END
#   NONE
#   PARAMETER_FAIL

GAP_TEMPLATE = r"""
LoadPackage("smallgrp");;

n      := {n};;
k      := {k};;
t      := {t};;
lambda := {lambda_};;
mu     := {mu};;

# Necessary feasibility condition: k(k - lambda - 1) = (n - k - 1) * mu
if k * (k - lambda - 1) <> (n - k - 1) * mu then
    Print("PARAMETER_FAIL\n");
    QUIT;
fi;

groups    := Filtered(AllSmallGroups(n), G -> not IsAbelian(G));;
numGroups := Size(groups);;
Print("GROUPS ", numGroups, "\n");

J       := List([1..n], i -> List([1..n], j -> 1));;
I       := IdentityMat(n);;
numSets := Binomial(n - 1, k);;

# Emit a SET token roughly every 1% so Python gets smooth progress updates
# without being flooded with output.
step := Maximum(1, Int(numSets / 100));;

group_index := 0;;

for G in groups do
    group_index := group_index + 1;
    Print("GROUP ", group_index, " ", numGroups, " ",
          StructureDescription(G), "\n");

    elements := AsList(G);;
    id       := Identity(G);;
    nonId    := Filtered(elements, x -> x <> id);;

    set_counter := 0;;

    for S in Combinations(nonId, k) do
        set_counter := set_counter + 1;

        if set_counter mod step = 0 then
            Print("SET ", set_counter, " ", numSets, "\n");
        fi;

        A := List([1..n], i -> List([1..n], j -> 0));;

        for row in [1..n] do
            g := elements[row];
            for s in S do
                h   := g * s;
                col := Position(elements, h);
                A[row][col] := 1;
            od;
        od;

        if A * A = t * I + lambda * A + mu * (J - I - A) then
            Print("FOUND\n");
            Print("ADJ_START\n");
            for row in A do
                Print(JoinStringsWithSeparator(List(row, String), ","), "\n");
            od;
            Print("ADJ_END\n");
            QUIT;
        fi;

    od;
od;

Print("NONE\n");
QUIT;
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _save_adjacency(
    adjacency: list[list[int]],
    n: int, k: int, t: int, lambda_: int, mu: int,
    logger: logging.Logger,
) -> str:
    filename = f"adjacency_{n}_{k}_{t}_{lambda_}_{mu}.json"
    Path(filename).write_text(json.dumps(adjacency, indent=2))
    logger.info(f"Adjacency matrix saved to {filename}")
    return filename


# ─────────────────────────────────────────────────────────────────────────────
# GAP runner
# ─────────────────────────────────────────────────────────────────────────────

def run_gap(
    n: int,
    k: int,
    t: int,
    lambda_: int,
    mu: int,
    logger: logging.Logger,
    progress: ProgressLine,
    timeout: float = 3600,
) -> dict:
    code = GAP_TEMPLATE.format(n=n, k=k, t=t, lambda_=lambda_, mu=mu)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".g", delete=False
    ) as f:
        f.write(code)
        script_path = f.name

    logger.debug(f"GAP script written to {script_path}")
    logger.info(
        f"Starting search: DSRG(n={n}, k={k}, t={t}, λ={lambda_}, μ={mu})"
    )

    start = time.perf_counter()

    proc = subprocess.Popen(
        ["gap", "-q", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    logger.info(f"GAP process started (PID {proc.pid})")

    # ── Drain stderr in a background thread ──────────────────────────────────
    # This prevents the stderr pipe buffer from filling up and blocking GAP.
    # All stderr output is surfaced in the log at DEBUG level.
    def _drain_stderr() -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.strip()
            if line:
                logger.debug(f"[GAP stderr] {line}")

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    # ── State tracked across lines ────────────────────────────────────────────
    num_groups  = 0
    group_idx   = 0
    group_name  = ""
    set_current = 0
    set_total   = comb(n - 1, k)
    adjacency: list[list[int]] = []
    capturing   = False

    def redraw() -> None:
        if num_groups == 0:
            return
        elapsed = _fmt_elapsed(time.perf_counter() - start)
        pct     = f"{100 * set_current / set_total:.1f}%" if set_total else "?"
        progress.update(
            f"[{elapsed}]  "
            f"Group {group_idx}/{num_groups}  ({group_name})  │  "
            f"Set {set_current:,}/{set_total:,}  ({pct})"
        )

    # ── Stream stdout ─────────────────────────────────────────────────────────
    assert proc.stdout is not None
    try:
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("GROUPS"):
                num_groups = int(line.split()[1])
                logger.info(f"{num_groups} nonabelian group(s) of order {n} to search")
                redraw()

            elif line.startswith("GROUP"):
                parts       = line.split(maxsplit=3)
                group_idx   = int(parts[1])
                group_name  = parts[3] if len(parts) > 3 else "?"
                set_current = 0
                logger.info(
                    f"Entering group {group_idx}/{num_groups}: {group_name}  "
                    f"({set_total:,} connection sets to check)"
                )
                redraw()

            elif line.startswith("SET"):
                parts       = line.split()
                set_current = int(parts[1])
                set_total   = int(parts[2])
                redraw()

            elif line == "ADJ_START":
                capturing = True
                adjacency = []

            elif line == "ADJ_END":
                capturing = False
                progress.finish()
                elapsed  = time.perf_counter() - start
                filename = _save_adjacency(adjacency, n, k, t, lambda_, mu, logger)
                logger.info(f"Total elapsed: {_fmt_elapsed(elapsed)}")
                proc.kill()
                stderr_thread.join(timeout=2)
                return {
                    "status":    "found",
                    "file":      filename,
                    "adjacency": adjacency,
                }

            elif capturing:
                adjacency.append([int(x) for x in line.split(",")])

            elif line == "FOUND":
                logger.info("✅  Match found — reading adjacency matrix…")

            elif line == "PARAMETER_FAIL":
                progress.finish()
                logger.warning(
                    f"Parameters (n={n}, k={k}, t={t}, λ={lambda_}, μ={mu}) "
                    f"fail the necessary condition k(k−λ−1) = (n−k−1)μ"
                )
                proc.kill()
                elapsed = time.perf_counter() - start
                logger.info(f"Total elapsed: {_fmt_elapsed(elapsed)}")
                stderr_thread.join(timeout=2)
                return {"status": "invalid_parameters"}

            elif line == "NONE":
                progress.finish()
                logger.info("Exhaustive search complete — no Cayley DSRG found")
                elapsed = time.perf_counter() - start
                logger.info(f"Total elapsed: {_fmt_elapsed(elapsed)}")
                stderr_thread.join(timeout=2)
                return {"status": "none_found"}

            else:
                # Unrecognised output from GAP — log it so nothing is silently lost.
                logger.debug(f"[GAP stdout] {line}")

        proc.wait(timeout=timeout)
        stderr_thread.join(timeout=2)

    except subprocess.TimeoutExpired:
        proc.kill()
        progress.finish()
        elapsed = time.perf_counter() - start
        logger.warning(f"Search timed out after {_fmt_elapsed(elapsed)}")
        stderr_thread.join(timeout=2)
        return {"status": "timeout"}

    progress.finish()
    elapsed = time.perf_counter() - start
    logger.info(f"Total elapsed: {_fmt_elapsed(elapsed)}")
    return {"status": "completed"}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Search for a Cayley-graph realisation of a DSRG via GAP.",
    )
    parser.add_argument("n",       type=int, help="Number of vertices")
    parser.add_argument("k",       type=int, help="Degree (in- and out-)")
    parser.add_argument("t",       type=int, help="Number of reciprocal neighbours")
    parser.add_argument("lambda_", type=int, metavar="lambda", help="λ parameter")
    parser.add_argument("mu",      type=int,                   help="μ parameter")
    parser.add_argument(
        "--timeout", type=float, default=3600,
        help="Wall-clock limit in seconds (default: 3600)",
    )
    parser.add_argument(
        "--log", type=Path, default=Path("cayley_search.log"),
        help="Log file path (default: cayley_search.log)",
    )
    args = parser.parse_args()

    progress = ProgressLine()
    logger   = setup_logger(args.log, progress)

    result = run_gap(
        args.n, args.k, args.t, args.lambda_, args.mu,
        logger   = logger,
        progress = progress,
        timeout  = args.timeout,
    )

    print("Result:", result)
