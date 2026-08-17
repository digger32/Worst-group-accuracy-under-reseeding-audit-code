#!/usr/bin/env python3
r"""Bind every \cite to a verified bib entry, and fail if one is not.

Wrong attribution is the failure mode that survives every other check: a fabricated
author-year reads perfectly and is only caught by a reader who knows the field. This
script refuses to let a citation whose metadata is still flagged reach a compiled
PDF, and reports which entries have been read rather than merely looked up.

    python scripts/cite_audit.py latex/a5_main.tex latex/a5.bib
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FLAGS = ("METADATA UNVERIFIED", "PLACEHOLDER")


def main(tex, bib):
    t, b = Path(tex).read_text(), Path(bib).read_text()
    entries = {m.group(1): m.group(2)
               for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", b, re.S)}
    cited = set()
    for c in re.findall(r"\\cite\{([^}]+)\}", t):
        cited |= {x.strip() for x in c.split(",")}

    missing = sorted(cited - set(entries))
    unverified = sorted(k for k in cited if k in entries
                        and any(f in entries[k] for f in FLAGS))
    read_pending = sorted(k for k in cited
                          if re.search(rf"% READ PDF[^@]*?@\w+\{{{re.escape(k)},", b, re.S))

    print(f"cited: {len(cited)}   in bib: {len(entries)}")
    if missing:
        print("MISSING from bib:", missing)
    if unverified:
        print("cited with UNVERIFIED metadata:", unverified)
    if read_pending:
        print(f"metadata verified, PDF not yet read ({len(read_pending)}):")
        for k in read_pending:
            print("   ", k)
    uncited = sorted(set(entries) - cited)
    if uncited:
        print("in bib but not cited:", uncited)

    if missing or unverified:
        print("\n[cite] FAIL -- resolve before compiling for submission")
        return 1
    print("\n[cite] PASS -- every citation binds to a verified entry")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "latex/a5_main.tex",
                  sys.argv[2] if len(sys.argv) > 2 else "latex/a5.bib"))
