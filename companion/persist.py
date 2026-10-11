"""Restart-safe append-only journal (JSON lines).

Every append is flushed and fsynced. On read, a torn or unparseable line
(a crash mid-write) is skipped and counted, never fatal, and never lowers
the high-water seq computed from the good lines.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple


class Journal:
    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        # Guarantee each record starts on a fresh line even after a torn tail.
        needs_newline = False
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                needs_newline = f.read(1) != b"\n"
        with open(self.path, "a", encoding="utf-8") as f:
            if needs_newline:
                f.write("\n")
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read(self) -> Tuple[List[Dict[str, Any]], int]:
        """Return (records, skipped_line_count)."""
        if not os.path.exists(self.path):
            return [], 0
        records: List[Dict[str, Any]] = []
        skipped = 0
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except ValueError:
                    skipped += 1
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
                else:
                    skipped += 1
        return records, skipped

    def max_seq(self, rec_type: str, issuer: Optional[str] = None) -> int:
        """Highest seq among records of rec_type (optionally for one issuer); -1 if none."""
        best = -1
        records, _ = self.read()
        for rec in records:
            if rec.get("type") != rec_type:
                continue
            if issuer is not None and rec.get("issuer") != issuer:
                continue
            seq = rec.get("seq")
            if isinstance(seq, int) and not isinstance(seq, bool) and seq > best:
                best = seq
        return best

    def nonces(self, rec_type: str) -> List[str]:
        records, _ = self.read()
        return [r["nonce"] for r in records
                if r.get("type") == rec_type and isinstance(r.get("nonce"), str)]
