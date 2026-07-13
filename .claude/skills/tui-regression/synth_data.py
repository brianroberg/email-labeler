"""Diverse synthetic data for the TUI e2e regression harness.

Produces the three record types the four TUIs consume, deliberately spanning
degenerate / typical / stress-edge cases so a full-workflow drive exercises the
whole surface (empty bodies, many stories, wide/emoji chars, CRLF, unlocatable
story text, every tier/theme/label, excluded/reviewed/labeled mixes).

Import the builders directly in a Pilot driver, or run this module to write the
JSONL files a real ``load_*`` path can read back:

    python -m evals ...   # (not this) — instead:
    python .claude/skills/tui-regression/synth_data.py <target_dir>
"""

import base64
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Repo modules (run from repo root).
from evals.newsletter_schemas import GoldenNewsletter, GoldenStory
from evals.schemas import GoldenThread

_THEMES = ["scripture", "christlikeness", "church", "vocation_family", "disciple_making"]


def _msg(body: str, headers=None) -> dict:
    """A Gmail-style message resource whose text/plain body decodes to *body*."""
    data = base64.urlsafe_b64encode(body.encode("utf-8")).decode("ascii")
    return {
        "payload": {
            "headers": headers or [{"name": "From", "value": "someone@example.com"}],
            "mimeType": "text/plain",
            "body": {"data": data},
        }
    }


def _scores(simple, concrete, personal, dynamic):
    return {"simple": simple, "concrete": concrete, "personal": personal, "dynamic": dynamic}


def _days_before_now_iso(days: int) -> str:
    """A midday-UTC ``send_date`` *days* before the real current instant, as a
    UTC ISO string.

    The past-N-days date-filter scenarios compute their cutoff from the real
    clock (``_days_ago_cutoff``), so the "recent" records must track ``now`` or
    the harness would rot when the wall clock advances. Midday-UTC keeps the
    local calendar date within one day of the offset regardless of the reader's
    timezone, and the large offsets used (2/10 vs 45/200 days) leave a wide
    margin around the 30-day boundary."""
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.replace(hour=12, minute=0, second=0, microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Newsletters (for evals.newsletter_label) — the only TUI with a model seam
# ---------------------------------------------------------------------------

def newsletters() -> list[GoldenNewsletter]:
    nls: list[GoldenNewsletter] = []

    # 1. Empty body — "no body to select from" edge path.
    nls.append(GoldenNewsletter(
        thread_id="nl-empty", message_id="m-empty", sender="news@dm.org",
        subject="(empty body newsletter)", body="",
    ))

    # 2. Single-line body, unreviewed, no stories (populated by scenarios as needed).
    nls.append(GoldenNewsletter(
        thread_id="nl-single", message_id="m-single", sender="updates@dm.org",
        subject="One story", body="Sarah joined campus ministry as a freshman and found faith.",
    ))

    # 3. Multi-story body (blank-line separated), locates cleanly.
    nls.append(GoldenNewsletter(
        thread_id="nl-multi", message_id="m-multi", sender="team@dm.org",
        subject="Three stories this month",
        body=(
            "Alice served meals at the shelter every Friday this spring.\n"
            "\n"
            "Bob preached his first sermon to the youth group.\n"
            "\n"
            "Carol organized a scripture study for new believers."
        ),
    ))

    # 4. Wide/emoji characters — exercises display-width wrapping.
    nls.append(GoldenNewsletter(
        thread_id="nl-emoji", message_id="m-emoji", sender="celebrate@dm.org",
        subject="Celebration 🎉 report",
        body=(
            "🎉 We celebrated 12 baptisms this weekend at the lake retreat.\n"
            "\n"
            "日本語のミニストリー also launched — 東京 team sends greetings 🙏 to all."
        ),
    ))

    # 5. CRLF line endings in the body (Windows-origin newsletter).
    nls.append(GoldenNewsletter(
        thread_id="nl-crlf", message_id="m-crlf", sender="win@dm.org",
        subject="CRLF body",
        body="First story about outreach.\r\n\r\nSecond story about discipleship.",
    ))

    # 6. Reviewed + fully labeled (Y flag; X-precedence contrast in list row).
    reviewed = GoldenNewsletter(
        thread_id="nl-reviewed", message_id="m-rev", sender="done@dm.org",
        subject="Already reviewed", body="Dana mentored three interns over the summer.",
        reviewed=True,
    )
    s = GoldenStory(story_id="nl-reviewed:0", text="Dana mentored three interns over the summer.",
                    expected_scores=_scores(3, 3, 3, 3), expected_tier="excellent",
                    expected_themes={"disciple_making": "emphasized"}, reviewed=True)
    reviewed.stories = [s]
    nls.append(reviewed)

    # 7. Excluded newsletter (only visible with --include-excluded).
    excl = GoldenNewsletter(
        thread_id="nl-excluded", message_id="m-excl", sender="skip@dm.org",
        subject="Excluded newsletter", body="Some administrative announcement only.",
        excluded=True,
    )
    nls.append(excl)

    # 8. Pre-populated story whose text is NOT in the body -> text-edit fallback.
    unloc = GoldenNewsletter(
        thread_id="nl-unlocatable", message_id="m-unloc", sender="edit@dm.org",
        subject="Unlocatable story", body="Body line one.\nBody line two.",
    )
    unloc.stories = [GoldenStory(
        story_id="nl-unlocatable:0",
        text="A hand-edited multi-line story\nthat no longer appears verbatim in the body.",
    )]
    nls.append(unloc)

    # 9. Many stories (>9) -> number keys 1-9, n/p wraparound, paging.
    many_body = "\n\n".join(f"Story number {i} about ministry work in region {i}." for i in range(12))
    many = GoldenNewsletter(
        thread_id="nl-many", message_id="m-many", sender="lots@dm.org",
        subject="Twelve stories", body=many_body,
    )
    many.stories = [
        GoldenStory(story_id=f"nl-many:{i}", text=f"Story number {i} about ministry work in region {i}.")
        for i in range(12)
    ]
    nls.append(many)

    # 10. Multi-paragraph single story (blank line inside one story) + mixed labels.
    mixed = GoldenNewsletter(
        thread_id="nl-mixed", message_id="m-mixed", sender="mix@dm.org",
        subject="Mixed labeled/unlabeled",
        body=(
            "Paragraph one of the retreat story.\n"
            "\n"
            "Paragraph two continues the same retreat story.\n"
            "\n"
            "A separate short story about a baptism."
        ),
    )
    mixed.stories = [
        GoldenStory(
            story_id="nl-mixed:0",
            text="Paragraph one of the retreat story.\n\nParagraph two continues the same retreat story.",
            expected_scores=_scores(3, 3, 2, 2), expected_tier="good",
            expected_themes={"church": "present"}, reviewed=True),
        GoldenStory(story_id="nl-mixed:1", text="A separate short story about a baptism."),  # unlabeled
    ]
    nls.append(mixed)

    # 11. Long sender + subject -> list-row truncation.
    nls.append(GoldenNewsletter(
        thread_id="nl-long", message_id="m-long",
        sender="a-very-long-sender-address-for-column-truncation@some-ministry-domain.org",
        subject="A deliberately long subject line that must be truncated to fit the list column width",
        body="Short body with one story about the mission trip.",
    ))

    return nls


# ---------------------------------------------------------------------------
# Threads (for evals.review + evals.edit_tui) — no model seam
# ---------------------------------------------------------------------------

def threads() -> list[GoldenThread]:
    ths: list[GoldenThread] = []

    ths.append(GoldenThread(
        thread_id="th-person-fyi", messages=[_msg("Hi, just wanted to say thanks for lunch!")],
        senders=["friend@gmail.com"], subject="Thanks", snippet="Hi, just wanted...",
        expected_sender_type="person", expected_label="fyi",
    ))
    ths.append(GoldenThread(
        thread_id="th-service-nr",
        messages=[_msg("Your invoice #4821 is due. Please remit payment by Friday.")],
        senders=["billing@vendor.com"], subject="Invoice due", snippet="Your invoice...",
        expected_sender_type="service", expected_label="needs_response",
    ))
    ths.append(GoldenThread(
        thread_id="th-lowpri",
        messages=[_msg("MEGA SALE!!! 50% off everything this weekend only!")],
        senders=["deals@shop.com"], subject="Sale", snippet="MEGA SALE",
        expected_sender_type="service", expected_label="low_priority",
    ))
    ths.append(GoldenThread(
        thread_id="th-excluded", messages=[_msg("Automated bounce notice.")],
        senders=["mailer-daemon@mail.com"], subject="Bounce", snippet="Automated bounce",
        expected_sender_type="service", expected_label="low_priority", excluded=True,
    ))
    ths.append(GoldenThread(
        thread_id="th-notes", messages=[_msg("Can we reschedule our 1:1 to Thursday?")],
        senders=["colleague@work.com"], subject="1:1", snippet="Can we reschedule",
        expected_sender_type="person", expected_label="needs_response",
        notes="pre-existing reviewer note", reviewed=True,
    ))
    ths.append(GoldenThread(
        thread_id="th-multimsg",
        messages=[_msg("First message in the thread about the project."),
                  _msg("Second reply with more detail and a question at the end?"),
                  _msg("Third message wrapping things up.")],
        senders=["a@team.com", "b@team.com"], subject="Project thread", snippet="First message...",
        expected_sender_type="person", expected_label="fyi",
    ))
    ths.append(GoldenThread(
        thread_id="th-reviewed", messages=[_msg("Newsletter: this week at the church.")],
        senders=["news@church.org"], subject="Weekly", snippet="Newsletter",
        expected_sender_type="service", expected_label="fyi", reviewed=True,
    ))
    long_body = "\n".join(
        f"Line {i} of a long body used to exercise scrolling in the detail view." for i in range(40)
    )
    ths.append(GoldenThread(
        thread_id="th-longbody", messages=[_msg(long_body)],
        senders=["verbose@example.com"], subject="Long body", snippet="Line 0...",
        expected_sender_type="person", expected_label="fyi",
    ))

    return ths


# ---------------------------------------------------------------------------
# Assessment records (for newsletter_review) — read-only browser, no model seam
# ---------------------------------------------------------------------------

def _story_rec(text, scores=None, tier=None, themes=None, qcot="", tcot=""):
    # Scores are 1/2/3 (Poor/OK/Good); themes are graded dicts (issue #53).
    avg = round(sum(scores.values()) / len(scores), 2) if scores else None
    return {"text": text, "scores": scores, "average_score": avg, "tier": tier,
            "themes": {} if themes is None else themes, "quality_cot": qcot, "theme_cot": tcot}


def assessment_records() -> list[dict]:
    recs: list[dict] = []

    # a1/a2 are dynamically-RECENT (inside a past-30d window); a3/a4 are
    # dynamically-OLD (outside it). Dates track the real clock so the past-N-days
    # filter scenarios don't rot (issue #36).
    recs.append({
        "timestamp": "2026-01-05T10:00:00+00:00", "message_id": "a1", "thread_id": "a1",
        "send_date": _days_before_now_iso(2),
        "from": "newsletter@dm.org", "subject": "Excellent edition", "overall_tier": "excellent",
        "stories": [_story_rec("Sarah led 5 students to a scripture retreat this weekend.",
                               _scores(3, 3, 3, 3), "excellent",
                               {"scripture": "emphasized", "disciple_making": "present"},
                               qcot="Concrete names and numbers throughout.", tcot="Clear scripture focus.")],
    })
    recs.append({
        "timestamp": "2026-01-06T11:00:00+00:00", "message_id": "a2", "thread_id": "a2",
        "send_date": _days_before_now_iso(10),
        "from": "weekly@ministry.org", "subject": "Good update", "overall_tier": "good",
        "stories": [_story_rec("The church hosted a community dinner.", _scores(3, 3, 2, 2),
                               "good", {"church": "present"}, qcot="Solid but generic.",
                               tcot="Church theme.")],
    })
    recs.append({
        "timestamp": "2026-01-07T12:00:00+00:00", "message_id": "a3", "thread_id": "a3",
        "send_date": _days_before_now_iso(45),
        "from": "digest@dm.org", "subject": "Fair digest", "overall_tier": "fair",
        "stories": [_story_rec("Some vocation and family news was shared.", _scores(2, 2, 2, 2),
                               "fair", {"vocation_family": "present"})],
    })
    recs.append({
        "timestamp": "2026-01-08T13:00:00+00:00", "message_id": "a4", "thread_id": "a4",
        "send_date": _days_before_now_iso(200),
        "from": "bulletin@church.net", "subject": "Poor bulletin", "overall_tier": "poor",
        "stories": [_story_rec("Announcements and administrative items only.", _scores(1, 1, 1, 2),
                               "poor", {})],
    })
    # No-stories AND no-send_date record: list column renders "—" and it sorts
    # last; a positive since-filter always drops it (issue #36).
    recs.append({
        "timestamp": "2026-01-09T14:00:00+00:00", "message_id": "a5", "thread_id": "a5",
        "from": "empty@dm.org", "subject": "No stories here", "overall_tier": None, "stories": [],
    })
    # a6/a7/a8 carry FIXED midday-UTC send-dates (distinct calendar days) so the
    # sort-order / init-since scenarios assert exact dates deterministically.
    # Multi-theme, multi-story, distinct sender for the sender filter.
    recs.append({
        "timestamp": "2026-01-10T15:00:00+00:00", "message_id": "a6", "thread_id": "a6",
        "send_date": "2026-08-01T12:00:00+00:00",
        "from": "multi@partner-org.com", "subject": "Multi-theme edition", "overall_tier": "good",
        "stories": [
            _story_rec("Christ-like service and scripture memorization combined.", _scores(3, 2, 3, 2),
                       "good", {"christlikeness": "emphasized", "scripture": "present"},
                       qcot="A" * 400, tcot="B" * 400),
            _story_rec("Disciple-making across three campuses.", _scores(3, 2, 3, 2),
                       "good", {"disciple_making": "emphasized"}),
        ],
    })
    # Emoji/wide subject.
    recs.append({
        "timestamp": "2026-01-11T16:00:00+00:00", "message_id": "a7", "thread_id": "a7",
        "send_date": "2026-07-15T12:00:00+00:00",
        "from": "global@dm.org", "subject": "🌍 東京 global report", "overall_tier": "excellent",
        "stories": [_story_rec("東京 team baptized 8 new believers 🙏.", _scores(3, 3, 3, 3),
                               "excellent", {"church": "present", "disciple_making": "emphasized"})],
    })
    # Long CoT for detail scrolling.
    recs.append({
        "timestamp": "2026-01-12T17:00:00+00:00", "message_id": "a8", "thread_id": "a8",
        "send_date": "2026-06-01T12:00:00+00:00",
        "from": "verbose@dm.org", "subject": "Long reasoning", "overall_tier": "fair",
        "stories": [_story_rec("A modest story.", _scores(2, 2, 2, 2), "fair", {"scripture": "present"},
                               qcot="\n".join(f"reasoning line {i}" for i in range(30)),
                               tcot="\n".join(f"theme reasoning {i}" for i in range(30)))],
    })
    # Evening-UTC send that falls on the PREVIOUS calendar day west of UTC:
    # 01:00 UTC on 07-05 is 20:00 CDT on 07-04. The list column and since-filter
    # must show/compare the LOCAL date (2026-07-04), never the UTC slice
    # (2026-07-05) (issue #36 + local-tz fix).
    recs.append({
        "timestamp": "2026-07-05T02:00:00+00:00", "message_id": "nd-evening", "thread_id": "nd-evening",
        "send_date": "2026-07-05T01:00:00+00:00",
        "from": "evening@dm.org", "subject": "Evening send lands prior local day",
        "overall_tier": "good",
        "stories": [_story_rec("An evening-sent story about outreach.", _scores(3, 2, 2, 3),
                               "good", {"church": "present"})],
    })

    return recs


# ---------------------------------------------------------------------------
# File emitters (exercise the real load_* paths)
# ---------------------------------------------------------------------------

def write_all(target_dir) -> dict:
    d = Path(target_dir)
    d.mkdir(parents=True, exist_ok=True)
    nl_path = d / "newsletter_golden.jsonl"
    th_path = d / "thread_golden.jsonl"
    as_path = d / "assessments.jsonl"
    nl_path.write_text("".join(json.dumps(n.to_dict()) + "\n" for n in newsletters()))
    th_path.write_text("".join(json.dumps(t.to_dict()) + "\n" for t in threads()))
    as_path.write_text("".join(json.dumps(r) + "\n" for r in assessment_records()))
    return {"newsletters": nl_path, "threads": th_path, "assessments": as_path}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    paths = write_all(target)
    print("Wrote synthetic data:")
    print(f"  {len(newsletters())} newsletters -> {paths['newsletters']}")
    print(f"  {len(threads())} threads      -> {paths['threads']}")
    print(f"  {len(assessment_records())} assessments  -> {paths['assessments']}")
