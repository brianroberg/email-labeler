"""Verify the newsletter eval sub-suite is documented in the technical README.

The top-level ``README-technical.md`` is the project's structure/reference doc: it
lists the project's modules under ``## Project Structure`` and maps each source
module to its tests under ``## Test Coverage by Module``. The newsletter evaluation
harness adds modules under ``evals/`` (``newsletter_harvest``, ``newsletter_run``,
etc.); this guards against them — and their tests — being omitted from those
references as the sub-suite grows.

Expectations are DERIVED from disk rather than a hand-maintained list: each
``evals/newsletter_*.py`` module must appear in the Project Structure section, and
each must have a correspondingly-named ``tests/test_eval_<module>.py`` documented in
the Test Coverage section. A newly added newsletter eval module therefore fails this
test until it (and its test) are documented. Meta-tests without a source module
(e.g. ``test_eval_newsletter_cli_docs.py``) are intentionally not required here,
mirroring how the analogous ``test_eval_cli_docs.py`` is not listed either.

Patterned after ``test_env_var_docs.py``. Uses pytest-subtests so each missing file
is reported individually on failure.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
README_PATH = ROOT / "README-technical.md"
EVALS_DIR = ROOT / "evals"
TESTS_DIR = ROOT / "tests"


def _newsletter_eval_modules() -> list[str]:
    """Module filenames (e.g. 'newsletter_run.py') for the newsletter eval suite."""
    return sorted(p.name for p in EVALS_DIR.glob("newsletter_*.py"))


def _section(readme_text: str, heading: str) -> str:
    """Extract a '## <heading>' section from the README (up to the next '## ')."""
    match = re.search(
        rf"## {re.escape(heading)}.*?(?=\n## |\Z)", readme_text, re.DOTALL
    )
    return match.group(0) if match else ""


def test_newsletter_eval_modules_documented(subtests):
    readme_text = README_PATH.read_text()
    structure = _section(readme_text, "Project Structure")
    modules = _newsletter_eval_modules()

    assert modules, "No evals/newsletter_*.py modules found — scan may be broken"
    assert structure, "No '## Project Structure' section found in README-technical.md"

    for module in modules:
        with subtests.test(module=module):
            assert module in structure, (
                f"evals/{module} is not listed in README-technical.md's "
                f"'## Project Structure' section"
            )


def test_newsletter_eval_tests_documented(subtests):
    readme_text = README_PATH.read_text()
    coverage = _section(readme_text, "Test Coverage by Module")
    modules = _newsletter_eval_modules()

    assert modules, "No evals/newsletter_*.py modules found — scan may be broken"
    assert coverage, "No '## Test Coverage by Module' section found in README-technical.md"

    for module in modules:
        stem = module[: -len(".py")]  # e.g. 'newsletter_run'
        test_name = f"test_eval_{stem}.py"
        with subtests.test(test=test_name):
            assert (TESTS_DIR / test_name).exists(), (
                f"evals/{module} has no matching {test_name}"
            )
            assert test_name in coverage, (
                f"{test_name} is not documented in README-technical.md's "
                f"'## Test Coverage by Module' section"
            )
