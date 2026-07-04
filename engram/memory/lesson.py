"""Lesson: one curated, actionable insight the agent has learned.

A lesson is more than text: it carries a scope (how broadly it applies) and
utility statistics (how it has actually performed in practice). Utility is the
principled signal the curator uses to decide what to keep - a lesson must earn its
place by correlating with successful attempts, not merely by existing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config

# Scope values. task_type and user lessons use scope_key to say *which*
# task type or user they apply to; global lessons apply everywhere.
SCOPES = ("global", "task_type", "user")


@dataclass
class Lesson:
    """A single curated insight.

    Attributes:
        id: Stable identifier (e.g. "L0007").
        content: The actionable insight, phrased generally.
        scope: One of SCOPES.
        scope_key: The task type or user this lesson is scoped to ("" if global).
        embedding: Semantic vector of content (for retrieval / similarity).
        retrieved_count: Times this lesson was injected into an attempt's context.
        helped_count: Retrievals where the attempt then succeeded.
        hurt_count: Retrievals where the attempt then failed.
        provenance: Ids of attempts/lessons that created or merged into this one.
        created_step: Logical training step at creation (reproducible, not wall-clock).
        version: Bumped on each merge/consolidation.
        source: "reflection" or "consolidation".
    """

    id: str
    content: str
    scope: str = "global"
    scope_key: str = ""
    embedding: list[float] | None = None
    retrieved_count: int = 0
    helped_count: int = 0
    hurt_count: int = 0
    provenance: list[str] = field(default_factory=list)
    created_step: int = 0
    version: int = 1
    source: str = "reflection"

    # ---- utility ----------------------------------------------------------- #
    @property
    def utility(self) -> float:
        """Shrunk helped-minus-hurt estimate in (-1, 1), centered at 0.

        Uses a pseudo-count prior so a single observation cannot swing the score to
        an extreme; evidence must accumulate before a lesson looks strongly good or
        bad. This is the number pruning is based on.
        """
        denom = self.helped_count + self.hurt_count + config.UTILITY_PRIOR
        return (self.helped_count - self.hurt_count) / denom

    @property
    def prunable(self) -> bool:
        """True once the lesson has had a fair trial and failed to earn its place."""
        return (
            self.retrieved_count >= config.PRUNE_MIN_RETRIEVED
            and self.utility <= config.PRUNE_UTILITY_FLOOR
        )

    def record_retrieval(self, success: bool) -> None:
        """Update utility stats after being retrieved for an attempt."""
        self.retrieved_count += 1
        if success:
            self.helped_count += 1
        else:
            self.hurt_count += 1

    def applies_to(self, task_type: str, user: str = "") -> bool:
        """Return True if this lesson is in scope for the given task/user."""
        if self.scope == "global":
            return True
        if self.scope == "task_type":
            return self.scope_key == task_type
        if self.scope == "user":
            return self.scope_key == user
        return False

    # ---- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "scope": self.scope,
            "scope_key": self.scope_key,
            "embedding": self.embedding,
            "retrieved_count": self.retrieved_count,
            "helped_count": self.helped_count,
            "hurt_count": self.hurt_count,
            "provenance": list(self.provenance),
            "created_step": self.created_step,
            "version": self.version,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Lesson:
        return cls(**d)
