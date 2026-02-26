#!/usr/bin/env python3
"""Parallelized Cayley graph DSRG search.

Splits the connector-set search space into blocks of configurable size and
dispatches them across worker processes, each of which runs a parameterised
GAP subprocess.  Progress is reported via tqdm as blocks complete.

Usage::

    python -m ilp.cayley_search.search n k t lambda mu \
        [--block-size 100000] [--workers NUM_CPUS] \
        [--timeout 3600] [--log FILE] [--stop-on-first]
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from math import comb
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

from tqdm import tqdm

# ---------------------------------------------------------------------------
# GAP script templates
# ---------------------------------------------------------------------------

_WORKER_TEMPLATE = r"""
LoadPackage("smallgrp");;

n         := {n};;
k         := {k};;
t         := {t};;
lambda    := {lambda_};;
mu        := {mu};;
libId     := {group_lib_id};;
startIdx  := {start_idx};;
endIdx    := {end_idx};;

G        := SmallGroup(n, libId);;
elements := AsList(G);;
id       := Identity(G);;
nonId    := Filtered(elements, x -> x <> id);;

J := List([1..n], i -> List([1..n], j -> 1));;
I := IdentityMat(n);;

allSets := Combinations(nonId, k);;
slice   := allSets{{[startIdx..endIdx]}};;

Print("BLOCK_START ", startIdx, " ", endIdx, "\n");

ProcessBlock := function()
    local S, A, row, g, s, h, col, checked;
    checked := 0;
    for S in slice do
        checked := checked + 1;

        A := List([1..n], i -> List([1..n], j -> 0));
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
        fi;
    od;
    Print("BLOCK_DONE ", checked, "\n");
end;;

ProcessBlock();;
QUIT;
"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroupInfo:
    """Metadata for a single nonabelian group."""

    filtered_index: int
    library_id: int
    name: str


@dataclass(frozen=True, slots=True)
class Job:
    """One unit of work: a block of connector sets for a specific group."""

    n: int
    k: int
    t: int
    lambda_: int
    mu: int
    group_lib_id: int
    group_name: str
    start_idx: int
    end_idx: int
    timeout: float


@dataclass(slots=True)
class BlockResult:
    """Result returned by a single worker."""

    status: str
    group_lib_id: int
    group_name: str
    start_idx: int
    end_idx: int
    checked: int = 0
    adjacency: list[list[int]] | None = None


# ---------------------------------------------------------------------------
# Shared state for stop-on-first
# ---------------------------------------------------------------------------

_found_flag: mp.Value | None = None  # type: ignore[type-arg]


def _init_pool(flag: mp.Value) -> None:  # type: ignore[type-arg]
    """Pool initializer — stash the shared flag in a module global."""
    global _found_flag  # noqa: PLW0603
    _found_flag = flag


# ---------------------------------------------------------------------------
# Metadata phase
# ---------------------------------------------------------------------------


def _run_metadata(n: int, logger: logging.Logger) -> list[GroupInfo]:
    """Run metadata.g to enumerate nonabelian groups of order *n*.

    Returns:
        List of ``GroupInfo`` for each nonabelian group found.
    """
    script = Path(__file__).with_name("metadata.g")
    proc = subprocess.run(
        ["gap", "-q", str(script), str(n)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        logger.error("metadata.g failed:\n%s", proc.stderr)
        sys.exit(1)

    groups: list[GroupInfo] = []
    for line in proc.stdout.splitlines():
        if line.startswith("GROUP_COUNT"):
            count = int(line.split()[1])
            logger.info("%d nonabelian group(s) of order %d", count, n)
        elif line.startswith("GROUP "):
            parts = line.split(maxsplit=3)
            groups.append(
                GroupInfo(
                    filtered_index=int(parts[1]),
                    library_id=int(parts[2]),
                    name=parts[3] if len(parts) > 3 else "?",
                )
            )
        elif line == "META_DONE":
            break
    return groups


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _run_worker(job: Job) -> BlockResult:
    """Execute a single GAP block-check in a subprocess.

    Returns:
        A ``BlockResult`` with status, checked count, and optional adjacency.
    """
    if _found_flag is not None and _found_flag.value:
        return BlockResult(
            status="skipped",
            group_lib_id=job.group_lib_id,
            group_name=job.group_name,
            start_idx=job.start_idx,
            end_idx=job.end_idx,
        )

    code = _WORKER_TEMPLATE.format(
        n=job.n,
        k=job.k,
        t=job.t,
        lambda_=job.lambda_,
        mu=job.mu,
        group_lib_id=job.group_lib_id,
        start_idx=job.start_idx,
        end_idx=job.end_idx,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".g", delete=False, encoding="utf-8") as f:
        f.write(code)
        script_path = f.name

    try:
        proc = subprocess.Popen(
            ["gap", "-q", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Drain stderr so the pipe buffer never blocks GAP.
        def _drain() -> None:
            assert proc.stderr is not None
            for _ in proc.stderr:
                pass

        t = threading.Thread(target=_drain, daemon=True)
        t.start()

        adjacency: list[list[int]] = []
        capturing = False
        checked = 0
        found = False

        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("BLOCK_DONE"):
                checked = int(line.split()[1])
            elif line == "FOUND":
                found = True
            elif line == "ADJ_START":
                capturing = True
                adjacency = []
            elif line == "ADJ_END":
                capturing = False
            elif capturing:
                adjacency.append([int(x) for x in line.split(",")])

        proc.wait(timeout=job.timeout)
        t.join(timeout=2)

        if found and _found_flag is not None:
            _found_flag.value = 1

        return BlockResult(
            status="found" if found else "done",
            group_lib_id=job.group_lib_id,
            group_name=job.group_name,
            start_idx=job.start_idx,
            end_idx=job.end_idx,
            checked=checked,
            adjacency=adjacency if found else None,
        )

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return BlockResult(
            status="timeout",
            group_lib_id=job.group_lib_id,
            group_name=job.group_name,
            start_idx=job.start_idx,
            end_idx=job.end_idx,
        )
    except Exception as exc:
        proc.kill()
        proc.wait()
        return BlockResult(
            status=f"error: {exc}",
            group_lib_id=job.group_lib_id,
            group_name=job.group_name,
            start_idx=job.start_idx,
            end_idx=job.end_idx,
        )
    finally:
        Path(script_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    adj: list[list[int]],
    n: int,
    k: int,
    t: int,
    lambda_: int,
    mu: int,
    group_lib_id: int,
    logger: logging.Logger,
) -> Path:
    filename = Path(f"adjacency_{n}_{k}_{t}_{lambda_}_{mu}_g{group_lib_id}.json")
    filename.write_text(json.dumps(adj, indent=2), encoding="utf-8")
    logger.info("Adjacency matrix saved to %s", filename)
    return filename


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------


def search(
    n: int,
    k: int,
    t: int,
    lambda_: int,
    mu: int,
    *,
    block_size: int = 100_000,
    num_workers: int | None = None,
    timeout: float = 3600,
    logfile: Path = Path("cayley_search.log"),
    stop_on_first: bool = False,
) -> list[BlockResult]:
    """Run the parallelized Cayley DSRG search.

    Args:
        n: Number of vertices.
        k: Degree (in- and out-).
        t: Number of reciprocal neighbours.
        lambda_: Lambda DSRG parameter.
        mu: Mu DSRG parameter.
        block_size: Connector sets per job.
        num_workers: Worker processes (defaults to CPU count).
        timeout: Wall-clock limit in seconds.
        logfile: Path for the log file.
        stop_on_first: Stop after the first DSRG is found.

    Returns:
        List of BlockResult objects for every dispatched block.
    """
    # -- Logger ---------------------------------------------------------------
    logger = logging.getLogger("cayley_search")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # -- Feasibility ----------------------------------------------------------
    lhs = k * (k - lambda_) - t
    rhs = (n - k - 1) * mu
    if lhs != rhs:
        logger.error(
            "Parameters (n=%d, k=%d, t=%d, lambda=%d, mu=%d) fail the necessary condition "
            "k(k - lambda) - t = (n - k - 1) * mu  (%d != %d)",
            n, k, t, lambda_, mu, lhs, rhs,
        )
        return []

    # -- Metadata -------------------------------------------------------------
    start_time = time.perf_counter()
    logger.info("DSRG(%d, %d, %d, %d, %d) — starting search", n, k, t, lambda_, mu)

    groups = _run_metadata(n, logger)
    if not groups:
        logger.info("No nonabelian groups of order %d — nothing to search", n)
        return []

    sets_per_group = comb(n - 1, k)
    logger.info("%s connector sets per group (C(%d, %d))", f"{sets_per_group:,}", n - 1, k)

    # -- Job list -------------------------------------------------------------
    worker_timeout = max(timeout / (num_workers or mp.cpu_count()) * 2, 60)
    jobs: list[Job] = []
    for g in groups:
        for start in range(1, sets_per_group + 1, block_size):
            end = min(start + block_size - 1, sets_per_group)
            jobs.append(
                Job(
                    n=n, k=k, t=t, lambda_=lambda_, mu=mu,
                    group_lib_id=g.library_id,
                    group_name=g.name,
                    start_idx=start,
                    end_idx=end,
                    timeout=worker_timeout,
                )
            )

    total_blocks = len(jobs)
    workers = num_workers or mp.cpu_count()
    logger.info(
        "%d block(s) across %d group(s), %d worker(s), block size %s",
        total_blocks, len(groups), workers, f"{block_size:,}",
    )

    # -- Dispatch -------------------------------------------------------------
    flag = mp.Value("b", 0) if stop_on_first else None

    results: list[BlockResult] = []
    total_checked = 0
    found_count = 0

    with mp.Pool(
        processes=workers,
        initializer=_init_pool,
        initargs=(flag if flag is not None else mp.Value("b", 0),),
    ) as pool:
        imap_it = pool.imap_unordered(_run_worker, jobs, chunksize=1)
        try:
            for result in tqdm(imap_it, total=total_blocks, desc="Blocks", unit="blk"):
                results.append(result)
                total_checked += result.checked

                if result.status == "found":
                    found_count += 1
                    assert result.adjacency is not None
                    tqdm.write(
                        f"  DSRG found!  group={result.group_name} "
                        f"(lib_id={result.group_lib_id})  "
                        f"block=[{result.start_idx}..{result.end_idx}]"
                    )
                    _save_adjacency(
                        result.adjacency, n, k, t, lambda_, mu,
                        result.group_lib_id, logger,
                    )
                elif result.status not in {"done", "skipped"}:
                    tqdm.write(f"  Worker issue: {result.status}")

        except KeyboardInterrupt:
            logger.warning("Interrupted — terminating workers")
            pool.terminate()

    # -- Summary --------------------------------------------------------------
    elapsed = time.perf_counter() - start_time
    logger.info(
        "Search complete in %s — %s sets checked, %d DSRG(s) found",
        _fmt_elapsed(elapsed), f"{total_checked:,}", found_count,
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parallelized Cayley-graph DSRG search via GAP.",
    )
    parser.add_argument("n", type=int, help="Number of vertices")
    parser.add_argument("k", type=int, help="Degree (in- and out-)")
    parser.add_argument("t", type=int, help="Number of reciprocal neighbours")
    parser.add_argument("lambda_", type=int, metavar="lambda", help="lambda parameter")
    parser.add_argument("mu", type=int, help="mu parameter")
    parser.add_argument(
        "--block-size", type=int, default=100_000,
        help="Connector sets per worker block (default: 100000)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--timeout", type=float, default=3600,
        help="Wall-clock limit in seconds (default: 3600)",
    )
    parser.add_argument(
        "--log", type=Path, default=Path("cayley_search.log"),
        help="Log file path (default: cayley_search.log)",
    )
    parser.add_argument(
        "--stop-on-first", action="store_true",
        help="Stop after the first DSRG is found",
    )
    args = parser.parse_args()

    search(
        args.n, args.k, args.t, args.lambda_, args.mu,
        block_size=args.block_size,
        num_workers=args.workers,
        timeout=args.timeout,
        logfile=args.log,
        stop_on_first=args.stop_on_first,
    )
