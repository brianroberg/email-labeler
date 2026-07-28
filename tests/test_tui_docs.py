"""Verify that every TUI in the project is documented in a README section.

The project ships several Textual TUIs (the newsletter review browser, the eval
review/edit/label tools). A TUI that nobody can find is a TUI nobody uses, so
each one must be documented in the nearest human-facing ``README.md``:

- **Launchable TUIs** (``python -m <target>``) must have their launch command
  shown in a fenced code block inside some README section. A passing prose
  mention elsewhere in the README does not count — the point is that a reader
  scanning the README's sections finds a runnable command.
- **Non-launchable TUIs** (screens reached through another tool's flag, e.g.
  ``evals/edit_tui.py`` via ``evals.review --edit``) have no command of their
  own, so the floor is that the README refers to the module by name.

Expectations are DERIVED from disk, not from a hand-maintained list: the TUIs
are discovered by parsing every non-test ``*.py`` file and looking for a
``textual`` ``App`` subclass, and each one's documentation home is the nearest
``README.md`` walking up from its directory (``evals/README.md`` for eval tools,
the root ``README.md`` for everything else). A TUI added in the future therefore
fails these tests until it is documented — no test edit required.

Patterned after ``test_env_var_docs.py`` and ``test_newsletter_eval_docs.py``.
Uses pytest-subtests so each undocumented TUI is reported individually.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Directories that never hold shipped TUIs (tests exercise them; the rest is
# tooling, vendored code, or scratch output).
_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    "docs",
    "node_modules",
    "tests",
    "results",
}

_MAIN_GUARD = re.compile(r'^if __name__ == ["\']__main__["\']:', re.MULTILINE)


def _unwrap_base(node: ast.expr) -> str:
    """Best-effort name of a class base: ``App``, ``app.App``, ``App[str]`` -> 'App'."""
    if isinstance(node, ast.Subscript):
        return _unwrap_base(node.value)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _defines_textual_app(tree: ast.Module) -> bool:
    """True if the module imports from textual and subclasses its ``App``."""
    imports_textual = any(
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("textual")
        or isinstance(node, ast.Import)
        and any(alias.name.startswith("textual") for alias in node.names)
        for node in ast.walk(tree)
    )
    if not imports_textual:
        return False

    return any(
        isinstance(node, ast.ClassDef)
        and any(_unwrap_base(base) == "App" for base in node.bases)
        for node in ast.walk(tree)
    )


def _launch_target(path: Path, source: str) -> str | None:
    """The ``python -m <target>`` target for a TUI module, or None if it has none.

    A package with a ``__main__.py`` is launched as the package
    (``python -m newsletter_review``); a module with an ``if __name__ ==
    "__main__":`` guard is launched as itself (``python -m evals.review``).
    Anything else is reached through another tool and has no command.
    """
    package_dir = path.parent
    if (package_dir / "__main__.py").exists() and (package_dir / "__init__.py").exists():
        return ".".join(package_dir.relative_to(ROOT).parts)
    if _MAIN_GUARD.search(source):
        return ".".join(path.relative_to(ROOT).with_suffix("").parts)
    return None


def _nearest_readme(path: Path) -> Path | None:
    """The closest human-facing README.md at or above the module's directory."""
    directory = path.parent
    while True:
        candidate = directory / "README.md"
        if candidate.exists():
            return candidate
        if directory == ROOT:
            return None
        directory = directory.parent


def _discover_tuis() -> list[dict]:
    """Every shipped Textual TUI, with its launch target and documentation home."""
    tuis = []
    for path in sorted(ROOT.rglob("*.py")):
        if _SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        source = path.read_text()
        if "textual" not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:  # pragma: no cover - not our problem to report here
            continue
        if not _defines_textual_app(tree):
            continue
        tuis.append(
            {
                "path": path,
                "rel": path.relative_to(ROOT).as_posix(),
                "module": ".".join(path.relative_to(ROOT).with_suffix("").parts),
                "launch_target": _launch_target(path, source),
                "readme": _nearest_readme(path),
            }
        )
    return tuis


def _code_blocks_by_section(readme_text: str) -> list[tuple[str, str]]:
    """(heading, code block) pairs for every fenced block in the README."""
    blocks = []
    heading = "(top of file)"
    in_block = False
    fence = ""
    current: list[str] = []

    for line in readme_text.splitlines():
        if in_block:
            if line.startswith(fence):
                blocks.append((heading, "\n".join(current)))
                in_block = False
                current = []
            else:
                current.append(line)
            continue
        if line.startswith("```"):
            in_block = True
            fence = "```"
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip()

    return blocks


def test_tuis_discovered():
    """Guard the scan itself: silence here would make the tests below vacuous."""
    tuis = _discover_tuis()
    assert tuis, "No Textual TUIs found — the discovery scan may be broken"
    for tui in tuis:
        assert tui["readme"] is not None, (
            f"{tui['rel']} has no README.md at or above its directory"
        )


def test_launchable_tuis_have_a_documented_command(subtests):
    for tui in _discover_tuis():
        target = tui["launch_target"]
        if target is None:
            continue
        command = f"python -m {target}"
        readme = tui["readme"]
        documented = [
            heading
            for heading, block in _code_blocks_by_section(readme.read_text())
            if command in block
        ]
        with subtests.test(tui=tui["rel"]):
            assert documented, (
                f"{tui['rel']} is a TUI launched with `{command}`, but no section of "
                f"{readme.relative_to(ROOT).as_posix()} shows that command in a code "
                f"block. Add a section documenting how to run it."
            )


def test_non_launchable_tuis_are_named_in_the_readme(subtests):
    for tui in _discover_tuis():
        if tui["launch_target"] is not None:
            continue
        readme = tui["readme"]
        readme_text = readme.read_text()
        names = {tui["module"], tui["rel"], tui["path"].stem}
        with subtests.test(tui=tui["rel"]):
            assert any(name in readme_text for name in names), (
                f"{tui['rel']} is a TUI with no launch command of its own, and "
                f"{readme.relative_to(ROOT).as_posix()} never refers to it by name. "
                f"Document the tool it is reached through and name the module."
            )
