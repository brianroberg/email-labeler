"""Keep .env.example in sync with README-technical's env-var table.

Two decision-free directions (Wave 1 plan, T4):
(a) every variable the example declares (active or commented) must be
    documented in the README's Environment Variables table — catches a
    stale or misspelled example;
(b) every variable the table marks Required must appear as an *active*
    (uncommented) line — catches the example losing a var the daemon
    won't start without.

Deliberately not asserted: that every documented optional/override knob
appears in the example — what .env.example is *for* (minimal quickstart vs
exhaustive catalog) is an owner call no test should pre-empt.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
ENV_EXAMPLE_PATH = ROOT / ".env.example"
README_PATH = ROOT / "README-technical.md"

# A declared var: optionally commented ("# "), then NAME= at line start.
_DECLARED_VAR = re.compile(r"^#? ?([A-Z_]+)=", re.MULTILINE)
# An active (uncommented) var line.
_ACTIVE_VAR = re.compile(r"^([A-Z_]+)=", re.MULTILINE)
# A table row marked Required: | `VAR` | Yes | ...
_REQUIRED_ROW = re.compile(r"\| `(\w+)` \| Yes \|")


def _env_vars_section(readme_text: str) -> str:
    match = re.search(r"## Environment Variables.*?(?=\n## |\Z)", readme_text, re.DOTALL)
    return match.group(0) if match else ""


def test_env_example_vars_are_documented(subtests):
    example_text = ENV_EXAMPLE_PATH.read_text()
    env_section = _env_vars_section(README_PATH.read_text())
    declared = set(_DECLARED_VAR.findall(example_text))

    assert declared, "No variables found in .env.example — parse may be broken"
    assert env_section, "No '## Environment Variables' section found in README"

    for var in sorted(declared):
        with subtests.test(var=var):
            assert f"`{var}`" in env_section, (
                f"`{var}` is declared in .env.example but not documented in "
                f"README-technical's Environment Variables table"
            )


def test_required_vars_present_and_active(subtests):
    example_text = ENV_EXAMPLE_PATH.read_text()
    env_section = _env_vars_section(README_PATH.read_text())
    required = set(_REQUIRED_ROW.findall(env_section))
    active = set(_ACTIVE_VAR.findall(example_text))

    assert required, "No Required rows found in the env table — parse may be broken"

    for var in sorted(required):
        with subtests.test(var=var):
            assert var in active, (
                f"`{var}` is marked Required in README-technical's env table "
                f"but has no active (uncommented) line in .env.example"
            )
