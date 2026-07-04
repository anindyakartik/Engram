"""MemoryStore: the curated collection of lessons, with persistence + snapshots.

The store is intentionally dumb about policy (dedup, pruning, consolidation live
in curation.py); it owns identity, storage, and the logical clock used for
reproducible created_step values and per-checkpoint snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path

from engram.memory.lesson import Lesson


class MemoryStore:
    """An in-memory collection of lessons with JSONL persistence and snapshots."""

    def __init__(self) -> None:
        self._lessons: dict[str, Lesson] = {}
        self._id_counter = 0
        self.step = 0  # logical clock; the harness advances it per training task

    # ---- identity / clock -------------------------------------------------- #
    def next_id(self) -> str:
        self._id_counter += 1
        return f"L{self._id_counter:04d}"

    def advance(self) -> None:
        """Advance the logical clock by one training step."""
        self.step += 1

    # ---- CRUD -------------------------------------------------------------- #
    def add(self, lesson: Lesson) -> Lesson:
        self._lessons[lesson.id] = lesson
        return lesson

    def remove(self, lesson_id: str) -> None:
        self._lessons.pop(lesson_id, None)

    def get(self, lesson_id: str) -> Lesson | None:
        return self._lessons.get(lesson_id)

    def all(self) -> list[Lesson]:
        return list(self._lessons.values())

    def __len__(self) -> int:
        return len(self._lessons)

    # ---- persistence ------------------------------------------------------- #
    def persist(self, path: str | Path) -> None:
        """Write all lessons as JSONL (one lesson per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for lesson in self._lessons.values():
                f.write(json.dumps(lesson.to_dict(), sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> MemoryStore:
        """Load a store from a JSONL file, restoring the id counter."""
        store = cls()
        for line in Path(path).read_text().splitlines():
            if line.strip():
                lesson = Lesson.from_dict(json.loads(line))
                store._lessons[lesson.id] = lesson
                num = int(lesson.id.lstrip("L"))
                store._id_counter = max(store._id_counter, num)
        return store

    def snapshot(self, path: str | Path) -> dict:
        """Write a checkpoint snapshot (lessons + summary) and return the summary.

        Embeddings are omitted from snapshots to keep them small and readable; the
        snapshot is what the dashboard and inspection tools browse.
        """
        lessons = []
        for lesson in self._lessons.values():
            row = {k: v for k, v in lesson.to_dict().items() if k != "embedding"}
            row["utility"] = round(lesson.utility, 4)  # derived; handy for inspection
            lessons.append(row)
        summary = {
            "step": self.step,
            "count": len(self._lessons),
            "by_scope": self._count_by("scope"),
            "by_source": self._count_by("source"),
            "lessons": lessons,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    def _count_by(self, attr: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for lesson in self._lessons.values():
            key = getattr(lesson, attr)
            out[key] = out.get(key, 0) + 1
        return out
