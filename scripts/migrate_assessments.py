"""Migrate pre-#53 records in a newsletter assessments JSONL to the current scheme.

Issue #53 replaced the 1-5 dimension scores with the 3-value Poor/OK/Good rubric
and list-shaped story themes with theme->grade dicts, and the readers were then
made to reject the old shapes outright — on the premise that the affected data
would simply be regenerated. That premise holds for the eval golden set. It
cannot hold for the production assessments JSONL: it is append-only and is the
only copy of every grading ever made, so its pre-#53 history has to be converted
instead. Until it is, one old record at the top of the file makes the whole file
unopenable in the review TUI.

What the conversion can and cannot preserve:

* **Themes** convert exactly in meaning: the old list asserted a theme was
  present, with no emphasis judgment, so each entry becomes ``present``. (Nothing
  becomes ``emphasized``, so no migrated record implies a theme label — matching
  the labels those emails actually carry.)
* **Dimension scores** cannot round-trip: 5 values are being mapped onto 3.
  They are bucketed 1-2 -> Poor, 3 -> OK, 4-5 -> Good, and ``average_score`` is
  recomputed so it agrees with the labels the detail view renders.
* **Tiers are preserved verbatim**, never recomputed. The tier is what the grader
  concluded under the old rubric, and it is what was applied to the email as a
  Gmail label; deriving a new one from re-bucketed dimensions would leave the
  record disagreeing with the label on the message.
* Migrated records are stamped ``"migrated_from": "pre-#53"`` so their Poor/OK/Good
  values are recognizable as re-bucketed 5-point judgments, and
  ``"schema_version": 1`` (decision D12) — the documented shape they are converted
  *to*. New-scheme records pass through byte-identical, unstamped.

Stop the daemon before migrating in place — it appends to this file.

Usage::

    python -m scripts.migrate_assessments data/newsletter_assessments.jsonl
    python -m scripts.migrate_assessments data/newsletter_assessments.jsonl --in-place
    python -m scripts.migrate_assessments data/newsletter_assessments.jsonl --out new.jsonl
"""

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# Old 1-5 dimension score -> new 3-value rubric (1=Poor, 2=OK, 3=Good).
_SCORE_BUCKETS = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
_VALID_SCORES = (1, 2, 3)
MIGRATION_STAMP = "pre-#53"


@dataclass
class MigrationStats:
    """How many records were read, and how many needed converting."""

    total: int = 0
    migrated: int = 0


def is_old_scheme(record: dict) -> bool:
    """True when *record* was written before the #53 scheme change.

    Two independent markers, either of which is conclusive: a story whose
    ``themes`` is a list (the old grader wrote ``[]`` even for no themes, so the
    shape alone identifies it), or a dimension score outside 1-3. A record with
    no stories carries neither marker — and has nothing to convert.
    """
    for story in record.get("stories", []):
        if isinstance(story.get("themes"), list):
            return True
        scores = story.get("scores") or {}
        if any(value not in _VALID_SCORES for value in scores.values()):
            return True
    return False


def _migrate_story(story: dict) -> dict:
    migrated = dict(story)

    themes = story.get("themes")
    if isinstance(themes, list):
        migrated["themes"] = {theme: "present" for theme in themes}

    scores = story.get("scores")
    if scores:
        bucketed = {}
        for dim, value in scores.items():
            try:
                bucketed[dim] = _SCORE_BUCKETS[value]
            except (KeyError, TypeError):
                # Refuse rather than pass it through. A value the old rubric never
                # produced leaves the record permanently flagged old-scheme (still
                # outside 1-3), so the next run would re-bucket its already-correct
                # siblings, walking Good(3) -> OK(2) -> Poor(1); a non-numeric one
                # would instead raise TypeError from the average, which the CLI does
                # not catch. Abort with the value named, original untouched.
                raise ValueError(
                    f"story dimension {dim!r} has score {value!r}, which is not a "
                    f"1-5 value the pre-#53 rubric could produce — refusing to guess"
                ) from None
        migrated["scores"] = bucketed
        migrated["average_score"] = sum(bucketed.values()) / len(bucketed)

    return migrated


def migrate_record(record: dict) -> tuple[dict, bool]:
    """Return ``(record, changed)``; new-scheme records pass through untouched.

    Never mutates the input.
    """
    if not is_old_scheme(record):
        return record, False

    migrated = dict(record)
    migrated["stories"] = [_migrate_story(story) for story in record.get("stories", [])]
    migrated["migrated_from"] = MIGRATION_STAMP
    # D12: the conversion target is the documented v1 shape, so converted
    # records are stamped with that version — a frozen literal, deliberately
    # not shared with write_assessment's current version (which bumps; this
    # doesn't). Pass-throughs above stay byte-identical: their versioning, or
    # lack of it, is not the migration's business.
    migrated["schema_version"] = 1
    return migrated, True


def migrate_file(
    path: Path,
    *,
    in_place: bool = False,
    out: Path | None = None,
) -> MigrationStats:
    """Migrate the records in *path*, reporting what was found.

    Without ``in_place`` or ``out`` this only counts (a dry run). ``in_place``
    rewrites *path* after copying it to ``<path>.bak``. Every line is parsed and
    converted before anything is written, so a malformed file aborts with the
    original untouched — half-migrating the only copy of the grading history is
    not a recoverable state.

    An ``in_place`` run with nothing to migrate is a no-op: it neither rewrites
    the file nor touches the backup. Otherwise a second run (out of doubt, or
    from a script) would replace the only pre-migration copy of an append-only,
    irreplaceable file with the already-migrated content.
    """
    stats = MigrationStats()
    lines = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {lineno} is not valid JSON: {exc}") from None
            stats.total += 1
            try:
                migrated, changed = migrate_record(record)
            except ValueError as exc:
                raise ValueError(f"{path}: line {lineno}: {exc}") from None
            stats.migrated += changed
            lines.append(json.dumps(migrated) + "\n")

    destination = out if out is not None else (path if in_place else None)
    if destination is None or (in_place and out is None and not stats.migrated):
        return stats

    if in_place and out is None:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))

    # Write beside the destination, then swap: an interrupted write must not be
    # able to leave a truncated assessments file behind.
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text("".join(lines))
    os.replace(temp, destination)
    return stats


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate pre-#53 newsletter assessment records to the current scheme.",
    )
    parser.add_argument("path", type=Path, help="Assessments JSONL to migrate")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the file (a .bak copy is kept). Stop the daemon first — it appends here.",
    )
    group.add_argument("--out", type=Path, help="Write the migrated records to this path instead")
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"Error: file not found: {args.path}", file=sys.stderr)
        return 1

    try:
        stats = migrate_file(args.path, in_place=args.in_place, out=args.out)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    where = args.out or (args.path if args.in_place else None)
    print(f"{stats.total} records, {stats.migrated} pre-#53 record(s) to migrate")
    if where is None:
        print("Dry run — nothing written. Re-run with --in-place (or --out PATH) to apply.")
    elif args.in_place and not stats.migrated:
        # Say what actually happened: claiming a write (and a backup) that was
        # deliberately skipped would send the reader looking for a .bak that the
        # first, real run left — or that never existed.
        print("Already migrated — file and backup left untouched.")
    else:
        print(f"Wrote {where}")
        if args.in_place:
            print(f"Backup: {args.path.with_suffix(args.path.suffix + '.bak')}")
    return 0


if __name__ == "__main__":
    sys.exit(cli())
