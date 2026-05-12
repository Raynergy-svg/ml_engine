"""Tier 2 T7: Brain section editor + BriefingDocument parser.

Parses `.claude/brain/briefing.md` into a list of ``## ``-headed sections,
exposes per-section CRUD, and writes back via an atomic temp+rename. The
``save_to_path`` method consults ``brain_caps`` and refuses writes that would
exceed 1.5x the registered hard-cap for the target filename (preventing the
brain from being silently bloated past the runtime guard in the
self-improvement layer).

See ``docs/superpowers/plans/2026-05-12-tier2-cherry-picks.md`` Task 7 for
the design rationale and the test cases that pin the contract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.scanner.automation.brain_caps import caps as _brain_caps


_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class BriefingSection:
    """Single ``## Heading``-delimited section of a brain markdown file."""

    title: str
    body: str  # excludes the heading line


@dataclass
class BriefingDocument:
    """Parsed representation of a brain markdown file.

    The document is a sequence of ``BriefingSection`` objects plus an optional
    ``preamble`` for everything that precedes the first ``##`` heading (the
    ``# Title`` line and any intro prose).
    """

    sections: List[BriefingSection] = field(default_factory=list)
    preamble: str = ""

    @classmethod
    def from_path(cls, path: Path) -> "BriefingDocument":
        text = path.read_text(encoding="utf-8")
        return cls.from_text(text)

    @classmethod
    def from_text(cls, text: str) -> "BriefingDocument":
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return cls(preamble=text, sections=[])
        preamble = text[: matches[0].start()].rstrip() + (
            "\n" if matches[0].start() > 0 else ""
        )
        sections: List[BriefingSection] = []
        for i, m in enumerate(matches):
            title = m.group(1)
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].lstrip("\n").rstrip()
            sections.append(BriefingSection(title=title, body=body))
        return cls(preamble=preamble, sections=sections)

    def to_text(self) -> str:
        parts: List[str] = []
        if self.preamble.strip():
            parts.append(self.preamble.rstrip() + "\n\n")
        for s in self.sections:
            parts.append(f"## {s.title}\n\n{s.body}\n\n")
        return "".join(parts).rstrip() + "\n"

    def update_section(self, title: str, body: str) -> None:
        """Replace the body of an existing section, or append a new one."""
        for s in self.sections:
            if s.title == title:
                s.body = body.rstrip()
                return
        self.sections.append(BriefingSection(title=title, body=body.rstrip()))

    def save_to_path(
        self,
        path: Path,
        *,
        filename_for_cap: str = "briefing.md",
    ) -> None:
        """Atomic write with ``brain_caps`` pre-write validation.

        Refuses to write if the resulting text exceeds 1.5x the hard-cap
        registered in ``brain_caps`` for ``filename_for_cap``. This is a
        soft fence — the runtime brain-cap guard still trims at the registered
        hard-cap; the 1.5x ceiling here just prevents the editor from being
        used to wholesale obliterate the cap in one save.
        """
        text = self.to_text()
        cap_table = _brain_caps()
        if filename_for_cap in cap_table:
            hard_cap, _warn_ratio = cap_table[filename_for_cap]
            if len(text) > int(hard_cap * 1.50):
                raise ValueError(
                    f"Would write {len(text):,} chars > 1.5x hard cap "
                    f"{hard_cap:,} for {filename_for_cap}; trim sections first."
                )
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    def reset_to_template(self, template_path: Path, *, target: Path) -> None:
        """Overwrite ``target`` with the contents of ``template_path``."""
        text = template_path.read_text(encoding="utf-8")
        target.write_text(text, encoding="utf-8")
