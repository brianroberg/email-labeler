"""Keep .env.example in sync with README-technical's env-var table.

Two decision-free directions (Wave 1 plan, T4):
(a) every variable the example declares (active or commented) must have a
    row in the README's Environment Variables table — catches a stale or
    misspelled example;
(b) every variable the table marks Required must appear as an *active*
    (uncommented) line — catches the example losing a var the daemon
    won't start without.

Deliberately not asserted: that every documented optional/override knob
appears in the example — what .env.example is *for* (minimal quickstart vs
exhaustive catalog) is an owner call no test should pre-empt.
"""

import re
from pathlib import Path

from tests.test_env_var_docs import _get_env_vars_section

ROOT = Path(__file__).parent.parent
ENV_EXAMPLE_PATH = ROOT / ".env.example"
README_PATH = ROOT / "README-technical.md"

# A declared var: optional indentation, optionally commented, then NAME=.
# [A-Z0-9_] (not \w) everywhere: env var names here are upper snake case,
# and one character class across all four regexes keeps digit-bearing names
# (OAUTH2_...) from being parsed inconsistently.
_DECLARED_VAR = re.compile(r"^\s*#?\s*([A-Z0-9_]+)=", re.MULTILINE)
# An active (uncommented) var line.
_ACTIVE_VAR = re.compile(r"^\s*([A-Z0-9_]+)=", re.MULTILINE)
# Any table row for a var: | `VAR` | ... (whitespace-tolerant cell padding).
_TABLE_ROW = re.compile(r"^\|\s*`([A-Z0-9_]+)`\s*\|", re.MULTILINE)
# A table row marked Required: | `VAR` | Yes | ...
_REQUIRED_ROW = re.compile(r"^\|\s*`([A-Z0-9_]+)`\s*\|\s*Yes\s*\|", re.MULTILINE)


def test_env_example_vars_are_documented(subtests):
    example_text = ENV_EXAMPLE_PATH.read_text()
    env_section = _get_env_vars_section(README_PATH.read_text())
    declared = set(_DECLARED_VAR.findall(example_text))
    documented = set(_TABLE_ROW.findall(env_section))

    assert declared, "No variables found in .env.example — parse may be broken"
    assert documented, "No table rows found in the env table — parse may be broken"

    for var in sorted(declared):
        with subtests.test(var=var):
            assert var in documented, (
                f"`{var}` is declared in .env.example but has no row in "
                f"README-technical's Environment Variables table"
            )


def test_required_vars_present_and_active(subtests):
    example_text = ENV_EXAMPLE_PATH.read_text()
    env_section = _get_env_vars_section(README_PATH.read_text())
    required = set(_REQUIRED_ROW.findall(env_section))
    active = set(_ACTIVE_VAR.findall(example_text))

    assert required, "No Required rows found in the env table — parse may be broken"

    for var in sorted(required):
        with subtests.test(var=var):
            assert var in active, (
                f"`{var}` is marked Required in README-technical's env table "
                f"but has no active (uncommented) line in .env.example"
            )
