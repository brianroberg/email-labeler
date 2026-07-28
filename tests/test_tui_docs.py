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
from functools import cache
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Directories that never hold shipped TUIs (tests and the agent skills' E2E
# drivers exercise them; the rest is tooling, vendored code, or scratch output).
_SKIP_DIRS = {
    ".claude",
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
# Fenced code block delimiters: backticks or tildes, optionally indented.
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
# ATX heading: up to three leading spaces, 1-6 '#', then whitespace.
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")


def _unwrap_base(node: ast.expr) -> str:
    """Best-effort name of a class base: ``App``, ``app.App``, ``App[str]`` -> 'App'."""
    if isinstance(node, ast.Subscript):
        return _unwrap_base(node.value)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _textual_app_names(tree: ast.Module) -> set[str]:
    """Local names that refer to Textual's ``App`` class in this module.

    Always includes the bare ``App`` (so attribute forms like ``textual.app.App``
    and ``app.App`` match after ``_unwrap_base``), plus any alias introduced by
    ``from textual.app import App as SomethingElse``.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("textual"):
            names.add("App")
            names.update(alias.asname or alias.name for alias in node.names if alias.name == "App")
        elif isinstance(node, ast.Import) and any(
            alias.name.startswith("textual") for alias in node.names
        ):
            names.add("App")
    return names


def _defines_textual_app(tree: ast.Module) -> bool:
    """True if the module imports from textual and subclasses its ``App``."""
    app_names = _textual_app_names(tree)
    if not app_names:
        return False

    return any(
        isinstance(node, ast.ClassDef)
        and any(_unwrap_base(base) in app_names for base in node.bases)
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


@cache
def _discover_tuis() -> list[dict]:
    """Every shipped Textual TUI, with its launch target and documentation home.

    Cached: each test below needs the whole list, and the scan reads and parses
    every source file in the project.
    """
    tuis = []
    for path in sorted(ROOT.rglob("*.py")):
        if _SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        source = path.read_text(encoding="utf-8")
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
    """(heading, code block) pairs for every fenced block in the README.

    Handles the fence forms CommonMark allows and READMEs actually use: backtick
    or tilde, and indented (a block nested under a list item). A block left
    unterminated at end of file still counts — the command it shows is documented
    either way.
    """
    blocks = []
    heading = "(top of file)"
    fence = ""  # the opening fence marker; empty while outside a block
    current: list[str] = []

    for line in readme_text.splitlines():
        match = _FENCE.match(line)
        if fence:
            # Only a run of the same character, at least as long, closes the block.
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence):
                blocks.append((heading, "\n".join(current)))
                fence = ""
                current = []
            else:
                current.append(line)
            continue
        if match:
            fence = match[1]
            continue
        if _HEADING.match(line):
            heading = line.strip().lstrip("#").strip()

    if fence:
        blocks.append((heading, "\n".join(current)))

    return blocks


@cache
def _readme_text(readme: Path) -> str:
    return readme.read_text(encoding="utf-8")


@cache
def _readme_blocks(readme: Path) -> tuple[tuple[str, str], ...]:
    return tuple(_code_blocks_by_section(_readme_text(readme)))


# --- Unit tests for the scan itself -------------------------------------------------
#
# ``test_tuis_discovered`` below only proves the scan found *something*. These
# exercise the helpers against synthetic input, so a hole that silently drops a
# real TUI (and with it every assertion about that TUI) fails here instead of
# passing vacuously there.


def _detects(source: str) -> bool:
    return _defines_textual_app(ast.parse(source))


def test_detects_plain_textual_app_subclass():
    assert _detects("from textual.app import App\nclass Mine(App): pass\n")


def test_detects_dotted_and_subscripted_app_bases():
    assert _detects("import textual.app\nclass Mine(textual.app.App): pass\n")
    assert _detects("from textual.app import App\nclass Mine(App[str]): pass\n")


def test_detects_aliased_app_import():
    """``import App as X`` is still Textual's App — the TUI must not slip through."""
    assert _detects("from textual.app import App as TextualApp\nclass Mine(TextualApp): pass\n")


def test_detects_subclass_of_a_module_local_app():
    """A TUI deriving from an intermediate base in the same module still counts."""
    assert _detects(
        "from textual.app import App\nclass Base(App): pass\nclass Mine(Base): pass\n"
    )


def test_ignores_modules_that_only_import_textual():
    assert not _detects("from textual.widgets import Static\nclass Mine(Static): pass\n")


def test_ignores_an_app_that_is_not_textuals():
    assert not _detects("from flask import App\nclass Mine(App): pass\n")


def test_code_blocks_are_found_when_the_fence_is_indented():
    """Fences nested under a list item are indented — still documentation."""
    markdown = "# Heading\n\n- run it:\n\n  ```bash\n  python -m mytool\n  ```\n"
    assert [block for _, block in _code_blocks_by_section(markdown) if "python -m mytool" in block]


def test_code_blocks_support_tilde_fences():
    markdown = "# Heading\n\n~~~bash\npython -m mytool\n~~~\n"
    assert [block for _, block in _code_blocks_by_section(markdown) if "python -m mytool" in block]


def test_code_blocks_survive_a_missing_closing_fence():
    """A README ending mid-block still documents the command it shows."""
    markdown = "# Heading\n\n```bash\npython -m mytool\n"
    assert [block for _, block in _code_blocks_by_section(markdown) if "python -m mytool" in block]


def test_headings_inside_code_blocks_are_not_treated_as_headings():
    markdown = "# Real\n\n```bash\n# just a shell comment\npython -m mytool\n```\n"
    assert [heading for heading, _ in _code_blocks_by_section(markdown)] == ["Real"]


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
        with subtests.test(tui=tui["rel"]):
            assert readme is not None, f"{tui['rel']} has no README.md at or above it"
            assert any(command in block for _, block in _readme_blocks(readme)), (
                f"{tui['rel']} is a TUI launched with `{command}`, but no section of "
                f"{readme.relative_to(ROOT).as_posix()} shows that command in a code "
                f"block. Add a section documenting how to run it."
            )


def test_non_launchable_tuis_are_named_in_the_readme(subtests):
    for tui in _discover_tuis():
        if tui["launch_target"] is not None:
            continue
        readme = tui["readme"]
        # Dotted or path form only — a bare stem like "report" would match
        # incidental prose and make this assertion vacuous.
        names = {tui["module"], tui["rel"]}
        with subtests.test(tui=tui["rel"]):
            assert readme is not None, f"{tui['rel']} has no README.md at or above it"
            assert any(name in _readme_text(readme) for name in names), (
                f"{tui['rel']} is a TUI with no launch command of its own, and "
                f"{readme.relative_to(ROOT).as_posix()} never refers to it by name. "
                f"Document the tool it is reached through and name the module."
            )
