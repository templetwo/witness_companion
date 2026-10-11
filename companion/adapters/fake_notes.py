"""Approved-note retrieval from a small JSON file that survives restart.

File shape: {"notes": [{"id": str, "text": str, "tags": [str], "approved": bool}]}
Only approved notes are ever returned. Writes are atomic (temp file + replace).
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

from .base import BackendUnavailable, Note, Notes


class FakeNotes(Notes):
    def __init__(self, path: str) -> None:
        self.path = path

    def _load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except ValueError as exc:
            raise BackendUnavailable("notes file unreadable: %s" % exc)
        notes = data.get("notes", []) if isinstance(data, dict) else []
        return [n for n in notes if isinstance(n, dict)]

    def add(self, note_id: str, text: str, tags: List[str], approved: bool) -> None:
        notes = [n for n in self._load() if n.get("id") != note_id]
        notes.append({"id": note_id, "text": text, "tags": list(tags), "approved": bool(approved)})
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent, prefix=".notes-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"notes": notes}, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def get(self, query: str) -> List[Note]:
        q = query.lower().strip()
        out = []
        for n in self._load():
            if n.get("approved") is not True:
                continue
            text = str(n.get("text", ""))
            tags = [str(t) for t in n.get("tags", []) if isinstance(t, str)]
            if not q or q in text.lower() or q in [t.lower() for t in tags]:
                out.append(Note(id=str(n.get("id", "")), text=text, tags=tags))
        return out
