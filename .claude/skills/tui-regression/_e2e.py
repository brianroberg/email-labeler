"""Shared helpers for the Pilot-driven TUI e2e drivers.

Every driver is a plain module exposing ``scenarios()`` -> list of async
``(name, coro_fn)`` and is run by ``run_all.py``. Helpers here mirror the
patterns the repo's own Pilot tests use (``run_test(size=SIZE)``, widget-state
reads, ``workers.wait_for_complete()``), so drivers stay short and correct.
"""

import asyncio
import os
import sys
import traceback

# Make repo root + this skill dir importable regardless of how we're launched.
_SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SKILL_DIR, "..", "..", ".."))
for _p in (_REPO_ROOT, _SKILL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SIZE = (100, 30)


async def drain(app, pilot):
    """Let @work workers (seeding, prompts) finish, then settle the UI."""
    await app.workers.wait_for_complete()
    await pilot.pause()


def static_text(app, selector: str) -> str:
    from textual.widgets import Static
    return str(app.screen.query_one(selector, Static).render())


def screen_text(app) -> str:
    """All rendered Static + Label text on the active screen (for substring checks)."""
    from textual.widgets import Label, Static
    parts = [str(w.render()) for w in app.screen.query(Static)]
    parts += [str(w.render()) for w in app.screen.query(Label)]
    return "\n".join(parts)


class Check:
    """Collects assertions inside a scenario without aborting on the first failure."""

    def __init__(self):
        self.failures: list[str] = []

    def that(self, cond: bool, msg: str):
        if not cond:
            self.failures.append(msg)

    def eq(self, got, want, msg: str):
        if got != want:
            self.failures.append(f"{msg}: got {got!r} != want {want!r}")


async def run_scenarios(scenarios, timeout: float = 30.0) -> list[tuple[str, bool, str]]:
    """Run each async scenario(check) coroutine; capture pass/fail + detail.

    Each scenario is bounded by *timeout* so one hanging Pilot flow surfaces as a
    localized FAIL instead of wedging the whole run.
    """
    results = []
    for name, fn in scenarios:
        chk = Check()
        try:
            await asyncio.wait_for(fn(chk), timeout=timeout)
            if chk.failures:
                results.append((name, False, "; ".join(chk.failures)))
            else:
                results.append((name, True, ""))
        except asyncio.TimeoutError:
            results.append((name, False, f"TIMEOUT after {timeout}s (a Pilot flow hung)"))
        except Exception:
            results.append((name, False, "EXCEPTION:\n" + traceback.format_exc()))
    return results


def report(driver_name: str, results: list[tuple[str, bool, str]]) -> int:
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n=== {driver_name}: {passed}/{len(results)} scenarios passed ===")
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}")
        if not ok:
            for line in detail.splitlines():
                print(f"         {line}")
    return 0 if passed == len(results) else 1
