"""Prompt + tool-description registries, backed by markdown files in ``prompts/``.

Both axes are iterable per ``PLANS.md``: the agent under test pulls its
tool-description text and system prompt from per-ID markdown files. Run IDs
(``baseline``, ``v1``, ...) live in the inference trace so we can A/B between
them with McNemar's downstream.

Layout::

    prompts/
      tool_descriptions/<id>.md
      system_prompts/<id>.md

The file content (with trailing whitespace stripped) is the prompt text. An
empty file maps to ``""`` — that's how the ``baseline`` system prompt anchor
is expressed.

Adding a new variant:
    Drop a new markdown file at ``prompts/tool_descriptions/v2.md``, then run
    with ``--tool-description-id v2``.

Conventions:
- ``baseline`` is the anchor used in every comparison.
- The ``baseline`` system prompt is intentionally an empty file → ``""``.
- ``v1`` placeholders ship with TODO text; iterate freely there.

The exposed ``TOOL_DESCRIPTIONS`` and ``SYSTEM_PROMPTS`` objects implement the
read-only dict surface that the rest of the codebase already depends on
(``__getitem__``, ``__contains__``, ``__iter__``, ``sorted(...)``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class _MarkdownPromptRegistry:
    """Dict-like read-only view over a directory of ``<id>.md`` prompt files."""

    def __init__(self, subdir: str) -> None:
        self._dir = _PROMPTS_DIR / subdir

    def _path(self, prompt_id: str) -> Path:
        return self._dir / f"{prompt_id}.md"

    def __getitem__(self, prompt_id: str) -> str:
        path = self._path(prompt_id)
        if not path.exists():
            raise KeyError(
                f"No prompt file at {path}. Available IDs: {sorted(self)}"
            )
        # rstrip trailing whitespace/newlines but preserve internal formatting.
        return path.read_text(encoding="utf-8").rstrip()

    def __contains__(self, prompt_id: object) -> bool:
        if not isinstance(prompt_id, str):
            return False
        return self._path(prompt_id).exists()

    def __iter__(self) -> Iterator[str]:
        if not self._dir.exists():
            return iter(())
        return (p.stem for p in sorted(self._dir.glob("*.md")))

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        return f"<MarkdownPromptRegistry dir={self._dir} ids={sorted(self)}>"


TOOL_DESCRIPTIONS = _MarkdownPromptRegistry("tool_descriptions")
SYSTEM_PROMPTS = _MarkdownPromptRegistry("system_prompts")


__all__ = ["TOOL_DESCRIPTIONS", "SYSTEM_PROMPTS"]
