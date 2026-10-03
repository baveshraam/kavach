"""The documents cannot silently rot.

CLAUDE.md section 1 makes "every task ends with its documents updated" a rule. A rule needs a tripwire: this
file fails when a markdown file is not registered in the documents map, when a documented command or script
does not exist, when a relative link is broken, or when a placeholder was left in. It cannot check that the
prose is *true*; it can check that the things the prose points at are real.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", ".venv-clone", "node_modules", "data", "dist", ".superpowers", "__pycache__", "pretrained_models"}
#: Where references to modules that may no longer exist are expected (dated records, generated reports).
HISTORICAL = ("docs/superpowers/", "paper/")


def markdown_files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*.md"):
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts):  # tool folders (.pytest_cache ...)
            continue
        out.append(p)
    return sorted(out)


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


CLAUDE = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
MAP_TOKENS = re.findall(r"`([^`\n]+\.md)`", CLAUDE)


def test_claude_md_states_the_rule():
    assert "The documentation rule" in CLAUDE and "End-of-task checklist" in CLAUDE
    assert "Documents map" in CLAUDE


@pytest.mark.parametrize("path", markdown_files(), ids=rel)
def test_every_markdown_file_is_registered_in_the_documents_map(path):
    r = rel(path)
    assert any(fnmatch.fnmatch(r, tok) for tok in MAP_TOKENS), (
        f"{r} is not in the documents map in CLAUDE.md. Register it (CLAUDE.md section 2) so it is kept current."
    )


def test_every_literal_path_in_the_map_exists():
    for tok in MAP_TOKENS:
        if any(ch in tok for ch in "*?["):
            assert any(fnmatch.fnmatch(rel(p), tok) for p in markdown_files()), f"the pattern {tok} matches nothing"
        else:
            assert (ROOT / tok).exists(), f"CLAUDE.md lists {tok}, which does not exist"


def living_docs() -> list[Path]:
    return [p for p in markdown_files() if not rel(p).startswith(HISTORICAL)]


@pytest.mark.parametrize("path", living_docs(), ids=rel)
def test_documented_commands_exist(path):
    text = path.read_text(encoding="utf-8")
    for mod in sorted(set(re.findall(r"python(?:\.exe)? -m (kavach(?:\.[A-Za-z0-9_]+)+)", text))):
        parts = mod.split(".")
        base = ROOT / "backend" / Path(*parts)
        assert base.with_suffix(".py").exists() or (base / "__init__.py").exists(), (
            f"{rel(path)} documents `python -m {mod}`, but backend/{'/'.join(parts)}.py does not exist"
        )
    for script in sorted(set(re.findall(r"\b(run_[a-z_]+\.ps1)\b", text))):
        assert (ROOT / script).exists(), f"{rel(path)} mentions {script}, which does not exist"


LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


@pytest.mark.parametrize("path", living_docs(), ids=rel)
def test_relative_links_resolve(path):
    text = path.read_text(encoding="utf-8")
    for target in LINK.findall(text):
        if re.match(r"^(https?:|mailto:|#)", target):
            continue
        file_part = target.split("#", 1)[0]
        if not file_part:
            continue
        assert (path.parent / file_part).exists(), f"{rel(path)} links to {target}, which does not exist"


@pytest.mark.parametrize("path", [p for p in markdown_files() if p.name != "CLAUDE.md"], ids=rel)  # CLAUDE.md names it
def test_no_placeholder_was_left_in(path):
    assert "TESTCOUNT" not in path.read_text(encoding="utf-8"), (
        f"{rel(path)} still contains TESTCOUNT: replace it with the current number (CLAUDE.md checklist item 5)"
    )
