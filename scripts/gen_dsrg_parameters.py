#!/usr/bin/env python3
"""Regenerate dsrg_parameters.csv from Brouwer's authoritative table.

Source: https://homepages.cwi.nl/~aeb/math/dsrg/dsrg.html

Status mapping from the comment column:
  - contains "does not exist" or "nonexistent"  -> impossible
  - empty or "?"                                -> open
  - otherwise (theorem refs like T1, M9, N12)   -> known
"""
import argparse
import csv
import re
import urllib.request
from html import unescape
from pathlib import Path

URL = "https://homepages.cwi.nl/~aeb/math/dsrg/dsrg.html"

TAG = re.compile(r"<[^>]+>")
TR = re.compile(r"<tr>(.*?)</tr>", re.S)
TD = re.compile(r"<td[^>]*>(.*?)</td>", re.S)


def strip(s):
    return TAG.sub("", unescape(s)).strip()


def classify(comment: str) -> str:
    c = comment.lower()
    if "does not exist" in c or "nonexistent" in c:
        return "impossible"
    if not comment.strip() or comment.strip() == "?":
        return "open"
    return "known"


def parse(html: str):
    """Parse Brouwer's tables. Complement rows have an empty first cell and
    inherit v from the preceding primary row."""
    rows = {}
    last_v = None
    for m in TR.finditer(html):
        cells = TD.findall(m.group(1))
        if len(cells) < 8:
            continue
        first = strip(cells[0])
        try:
            k, t, lam, mu = [int(strip(c)) for c in cells[1:5]]
        except ValueError:
            continue
        if first == "":
            if last_v is None:
                continue
            v = last_v
        else:
            try:
                v = int(first)
            except ValueError:
                continue
            last_v = v
        comment = strip(cells[7])
        key = (v, k, t, lam, mu)
        if key in rows:
            continue  # dedup: keep first occurrence
        rows[key] = comment
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/dsrg.html",
                    help="Local HTML cache; downloaded if missing")
    ap.add_argument("--output", default="dsrg_parameters.csv")
    ap.add_argument("--refresh", action="store_true",
                    help="Re-download even if cache exists")
    args = ap.parse_args()

    cache = Path(args.cache)
    if args.refresh or not cache.exists():
        print(f"Downloading {URL} -> {cache}")
        urllib.request.urlretrieve(URL, cache)
    html = cache.read_text()

    rows = parse(html)
    print(f"Parsed {len(rows)} parameter tuples")

    out = Path(args.output)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "k", "t", "lambda", "mu", "Status"])
        for (v, k, t, lam, mu) in sorted(rows):
            w.writerow([v, k, t, lam, mu, classify(rows[(v, k, t, lam, mu)])])

    from collections import Counter
    statuses = Counter(classify(c) for c in rows.values())
    print(f"Status breakdown: {dict(statuses)}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
