"""Tests for newsletter_review TUI — pure data functions + Pilot UI tests."""

import json

import pytest
from textual.widgets import Label, ListView, Static

from newsletter_review.tui import (
    DetailScreen,
    ReviewApp,
    apply_filters,
    build_detail_lines,
    format_filter_summary,
    format_list_row,
    load_assessments,
    run_review_tui,
    sort_by_send_date,
    wrap_text,
)


def _make_record(
    *,
    subject="February Update",
    sender="john@dm.org",
    overall_tier="good",
    stories=None,
    thread_id="t_001",
    message_id="msg_001",
    timestamp="2026-02-20T12:00:00Z",
    send_date="2026-02-19T09:00:00+00:00",
    model="claude-sonnet-4-6",
):
    if stories is None:
        stories = [
            {
                "text": "Once upon a time in a campus ministry...",
                "scores": {"simple": 4, "concrete": 3, "personal": 5, "dynamic": 2},
                "average_score": 3.5,
                "tier": "good",
                "themes": ["scripture", "church"],
                "quality_cot": "The story focuses on one idea.",
                "theme_cot": "This illustrates Scripture study.",
            }
        ]
    record = {
        "timestamp": timestamp,
        "message_id": message_id,
        "thread_id": thread_id,
        "from": sender,
        "subject": subject,
        "send_date": send_date,
        "model": model,
        "overall_tier": overall_tier,
        "stories": stories,
    }
    # Old records predate these fields; pass None to omit (backward-compat tests).
    if send_date is None:
        del record["send_date"]
    if model is None:
        del record["model"]
    return record


# ---------------------------------------------------------------------------
# load_assessments
# ---------------------------------------------------------------------------

class TestLoadAssessments:
    def test_loads_jsonl_records(self, tmp_path):
        f = tmp_path / "assessments.jsonl"
        r1 = _make_record(thread_id="t1")
        r2 = _make_record(thread_id="t2")
        f.write_text(json.dumps(r1) + "\n" + json.dumps(r2) + "\n")

        records = load_assessments(f)
        assert len(records) == 2
        assert records[0]["thread_id"] == "t1"
        assert records[1]["thread_id"] == "t2"

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "assessments.jsonl"
        r = _make_record()
        f.write_text("\n" + json.dumps(r) + "\n\n")

        records = load_assessments(f)
        assert len(records) == 1

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_assessments(tmp_path / "nonexistent.jsonl")

    def test_returns_empty_for_empty_file(self, tmp_path):
        f = tmp_path / "assessments.jsonl"
        f.write_text("")

        records = load_assessments(f)
        assert records == []


# ---------------------------------------------------------------------------
# apply_filters
# ---------------------------------------------------------------------------

class TestApplyFilters:
    def test_no_filters_returns_all(self):
        records = [_make_record(thread_id="t1"), _make_record(thread_id="t2")]
        result = apply_filters(records)
        assert len(result) == 2

    def test_tier_filter(self):
        records = [
            _make_record(overall_tier="good"),
            _make_record(overall_tier="poor"),
            _make_record(overall_tier="good"),
        ]
        result = apply_filters(records, tier="good")
        assert len(result) == 2

    def test_tier_filter_excludes_none_tier(self):
        records = [_make_record(overall_tier=None)]
        result = apply_filters(records, tier="good")
        assert result == []

    def test_theme_filter_matches_story_theme(self):
        records = [
            _make_record(stories=[{
                "text": "T", "scores": None,
                "average_score": None, "tier": None,
                "themes": ["scripture", "church"],
                "quality_cot": "", "theme_cot": "",
            }]),
            _make_record(stories=[{
                "text": "T", "scores": None,
                "average_score": None, "tier": None,
                "themes": ["disciple_making"],
                "quality_cot": "", "theme_cot": "",
            }]),
        ]
        result = apply_filters(records, theme="scripture")
        assert len(result) == 1

    def test_theme_filter_case_insensitive(self):
        records = [_make_record()]  # default has "scripture" theme
        result = apply_filters(records, theme="SCRIPTURE")
        assert len(result) == 1

    def test_sender_filter_substring_match(self):
        records = [
            _make_record(sender="john@dm.org"),
            _make_record(sender="jane@other.org"),
        ]
        result = apply_filters(records, sender="dm.org")
        assert len(result) == 1
        assert result[0]["from"] == "john@dm.org"

    def test_sender_filter_case_insensitive(self):
        records = [_make_record(sender="John@DM.org")]
        result = apply_filters(records, sender="john@dm.org")
        assert len(result) == 1

    def test_multiple_filters_are_anded(self):
        records = [
            _make_record(overall_tier="good", sender="john@dm.org"),
            _make_record(overall_tier="poor", sender="john@dm.org"),
            _make_record(overall_tier="good", sender="jane@other.org"),
        ]
        result = apply_filters(records, tier="good", sender="john")
        assert len(result) == 1

    def test_empty_input_returns_empty(self):
        result = apply_filters([], tier="good")
        assert result == []

    def test_since_filter_keeps_on_or_after_cutoff(self):
        # Issue #36: date filter on the send-date, inclusive of the boundary.
        records = [
            _make_record(thread_id="old", send_date="2024-01-01T00:00:00+00:00"),
            _make_record(thread_id="edge", send_date="2024-06-15T09:00:00+00:00"),
            _make_record(thread_id="new", send_date="2024-12-31T00:00:00+00:00"),
        ]
        result = apply_filters(records, since="2024-06-15")
        assert {r["thread_id"] for r in result} == {"edge", "new"}

    def test_since_filter_excludes_records_without_send_date(self):
        records = [
            _make_record(thread_id="dated", send_date="2024-12-31T00:00:00+00:00"),
            _make_record(thread_id="undated", send_date=None),
        ]
        result = apply_filters(records, since="2024-01-01")
        assert [r["thread_id"] for r in result] == ["dated"]


class TestSortBySendDate:
    def test_desc_newest_first(self):
        recs = [
            _make_record(thread_id="a", send_date="2024-01-01T00:00:00+00:00"),
            _make_record(thread_id="b", send_date="2024-03-01T00:00:00+00:00"),
            _make_record(thread_id="c", send_date="2024-02-01T00:00:00+00:00"),
        ]
        assert [r["thread_id"] for r in sort_by_send_date(recs)] == ["b", "c", "a"]

    def test_missing_send_date_sorts_last(self):
        recs = [
            _make_record(thread_id="undated", send_date=None),
            _make_record(thread_id="dated", send_date="2024-01-01T00:00:00+00:00"),
        ]
        assert [r["thread_id"] for r in sort_by_send_date(recs)] == ["dated", "undated"]


# ---------------------------------------------------------------------------
# _format_list_row
# ---------------------------------------------------------------------------

class TestFormatListRow:
    def test_includes_tier(self):
        row = format_list_row(_make_record(overall_tier="excellent"), 120)
        assert "excellent" in row

    def test_includes_sender(self):
        row = format_list_row(_make_record(sender="john@dm.org"), 120)
        assert "john@dm.org" in row

    def test_includes_story_count(self):
        record = _make_record()  # 1 story
        row = format_list_row(record, 120)
        assert "1" in row

    def test_truncates_long_subject(self):
        record = _make_record(subject="A" * 200)
        row = format_list_row(record, 80)
        assert "..." in row
        assert len(row) <= 80

    def test_handles_none_tier(self):
        row = format_list_row(_make_record(overall_tier=None), 120)
        # Should not crash; should show a placeholder
        assert row  # non-empty string

    def test_includes_send_date(self):
        # Issue #36: the list date column shows the email SEND date (date part).
        row = format_list_row(_make_record(send_date="2026-02-19T09:00:00+00:00"), 120)
        assert "2026-02-19" in row

    def test_missing_send_date_shows_placeholder(self):
        row = format_list_row(_make_record(send_date=None), 120)
        assert row.startswith("—")  # date column is first, placeholder for no send-date


# ---------------------------------------------------------------------------
# _build_detail_lines
# ---------------------------------------------------------------------------

class TestBuildDetailLines:
    def test_includes_subject_and_sender(self):
        lines = build_detail_lines(_make_record(subject="Feb Update", sender="john@dm.org"))
        text = "\n".join(lines)
        assert "Feb Update" in text
        assert "john@dm.org" in text

    def test_includes_overall_tier(self):
        lines = build_detail_lines(_make_record(overall_tier="excellent"))
        text = "\n".join(lines)
        assert "excellent" in text

    def test_header_shows_send_date_and_model_and_processed(self):
        # Issue #35: email send-date grouped with email-intrinsic data, and a
        # separate classification block with the processed date + model.
        lines = build_detail_lines(_make_record(
            send_date="2026-02-19T09:00:00+00:00",
            model="claude-sonnet-4-6",
            timestamp="2026-02-20T12:00:00+00:00",
        ))
        text = "\n".join(lines)
        assert "Sent:" in text
        assert "2026-02-19T09:00:00+00:00" in text  # send-date
        assert "Processed:" in text
        assert "2026-02-20T12:00:00+00:00" in text  # processed timestamp
        assert "Model:" in text
        assert "claude-sonnet-4-6" in text
        # The processed date must NOT be presented under a bare "Date:" label
        # (the exact #35 complaint).
        assert "Date:" not in text

    def test_header_missing_send_date_and_model_show_placeholders(self):
        lines = build_detail_lines(_make_record(send_date=None, model=None))
        text = "\n".join(lines)
        # Missing send-date must not be misrepresented as the processed date.
        assert "Sent: unknown" in text
        assert "Model: —" in text

    def test_includes_story_text_excerpt_and_tier(self):
        record = _make_record()
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        # Stories are identified by a text excerpt, not a title.
        assert "Once upon a time in a campus ministry" in text
        assert "good" in text

    def test_includes_quality_scores(self):
        record = _make_record()
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        assert "simple" in text.lower()
        assert "4" in text

    def test_includes_quality_cot(self):
        record = _make_record()
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        assert "The story focuses on one idea." in text

    def test_includes_theme_cot(self):
        record = _make_record()
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        assert "This illustrates Scripture study." in text

    def test_includes_themes(self):
        record = _make_record()
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        assert "scripture" in text

    def test_graded_themes_show_grade(self):
        # New records carry graded themes (theme -> present/emphasized); the
        # detail view shows the grade (issue #53).
        record = _make_record(stories=[{
            "text": "Content", "scores": None, "average_score": None, "tier": None,
            "themes": {"scripture": "emphasized", "church": "present"},
            "quality_cot": "", "theme_cot": "",
        }])
        text = "\n".join(build_detail_lines(record))
        assert "scripture (emphasized)" in text
        assert "church (present)" in text

    def test_legacy_list_themes_still_render(self):
        # Old records store a present-only list; still render as plain names.
        record = _make_record(stories=[{
            "text": "Content", "scores": None, "average_score": None, "tier": None,
            "themes": ["scripture", "church"], "quality_cot": "", "theme_cot": "",
        }])
        text = "\n".join(build_detail_lines(record))
        assert "scripture" in text
        assert "church" in text

    def test_handles_missing_scores(self):
        record = _make_record(stories=[{
            "text": "Content without any scores",
            "scores": None, "average_score": None, "tier": None,
            "themes": [], "quality_cot": "", "theme_cot": "",
        }])
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        assert "Content without any scores" in text

    def test_handles_multiple_stories(self):
        stories = [
            {
                "text": "Content A about a student",
                "scores": {"simple": 5, "concrete": 5, "personal": 5, "dynamic": 5},
                "average_score": 5.0, "tier": "excellent",
                "themes": ["scripture"], "quality_cot": "cot A", "theme_cot": "theme A",
            },
            {
                "text": "Content B about a mentor",
                "scores": {"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2},
                "average_score": 2.0, "tier": "fair",
                "themes": ["church"], "quality_cot": "cot B", "theme_cot": "theme B",
            },
        ]
        record = _make_record(stories=stories)
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        assert "Content A about a student" in text
        assert "Content B about a mentor" in text
        assert "1/2" in text
        assert "2/2" in text

    def test_handles_empty_stories_list(self):
        record = _make_record(stories=[])
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        assert "No stories" in text.lower() or "0" in text


# ---------------------------------------------------------------------------
# wrap_text
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# format_filter_summary
# ---------------------------------------------------------------------------

class TestFormatFilterSummary:
    def test_no_filters(self):
        assert format_filter_summary(tier=None, theme=None, sender=None) == ""

    def test_tier_only(self):
        result = format_filter_summary(tier="good", theme=None, sender=None)
        assert "tier:good" in result

    def test_theme_only(self):
        result = format_filter_summary(tier=None, theme="scripture", sender=None)
        assert "theme:scripture" in result

    def test_sender_only(self):
        result = format_filter_summary(tier=None, theme=None, sender="dm.org")
        assert "sender:dm.org" in result

    def test_all_filters(self):
        result = format_filter_summary(tier="poor", theme="church", sender="john")
        assert "tier:poor" in result
        assert "theme:church" in result
        assert "sender:john" in result

    def test_since_filter(self):
        result = format_filter_summary(since="2024-06-15")
        assert "since:2024-06-15" in result

    def test_returns_empty_for_all_none(self):
        assert format_filter_summary() == ""


# ---------------------------------------------------------------------------
# wrap_text
# ---------------------------------------------------------------------------

class TestWrapText:
    def test_wraps_long_line(self):
        text = "word " * 30  # ~150 chars
        lines = wrap_text(text, 40)
        assert all(len(line) <= 40 for line in lines)
        assert len(lines) > 1

    def test_preserves_existing_newlines(self):
        text = "line one\nline two\nline three"
        lines = wrap_text(text, 80)
        assert len(lines) >= 3

    def test_zero_width_returns_raw_lines(self):
        text = "hello\nworld"
        lines = wrap_text(text, 0)
        assert "hello" in lines
        assert "world" in lines

    def test_empty_string(self):
        lines = wrap_text("", 80)
        assert lines == [] or lines == [""]


# ---------------------------------------------------------------------------
# Pilot UI tests — drive the real Textual app: key presses in, widget state
# and rendered content out.
# ---------------------------------------------------------------------------

SIZE = (100, 30)


def _ui_records():
    return [
        _make_record(subject="Subject zero", overall_tier="good", sender="alice@dm.org"),
        _make_record(subject="Subject one", overall_tier="poor", sender="bob@other.org"),
        _make_record(subject="Subject two", overall_tier="good", sender="carol@dm.org"),
        _make_record(
            subject="Subject three",
            overall_tier="excellent",
            sender="dave@dm.org",
            stories=[{
                "text": "T", "scores": None,
                "average_score": None, "tier": None,
                "themes": ["vocation_family"], "quality_cot": "", "theme_cot": "",
            }],
        ),
        _make_record(subject="Subject four", overall_tier="fair", sender="erin@other.org"),
        _make_record(subject="Subject five", overall_tier="good", sender="frank@dm.org"),
    ]


def _title(app) -> str:
    return str(app.query_one("#title", Static).render())


class TestReviewAppList:
    async def test_list_shows_all_records(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE):
            assert len(app.query_one(ListView)) == 6
            assert "6 records" in _title(app)

    async def test_down_arrow_moves_cursor(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            assert app.query_one(ListView).index == 0
            await pilot.press("down")
            assert app.query_one(ListView).index == 1

    async def test_q_quits(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("q")
        assert app.return_value == "quit"


class TestReviewAppDetail:
    async def test_enter_opens_detail_with_content(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("down", "enter")
            assert isinstance(app.screen, DetailScreen)
            text = "\n".join(str(w.render()) for w in app.screen.query(Static))
            assert "Subject one" in text
            assert "Overall: poor" in text
            assert "Once upon a time in a campus ministry" in text

    async def test_detail_status_shows_position(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("down", "enter")
            text = "\n".join(str(w.render()) for w in app.screen.query(Static))
            assert "Newsletter 2/6" in text

    async def test_escape_returns_to_list_preserving_cursor(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("down", "down", "enter")
            assert isinstance(app.screen, DetailScreen)
            await pilot.press("escape")
            assert not isinstance(app.screen, DetailScreen)
            assert app.query_one(ListView).index == 2

    async def test_q_quits_from_detail(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            assert isinstance(app.screen, DetailScreen)
            await pilot.press("q")
        assert app.return_value == "quit"

    async def test_detail_scrolls_with_arrow_keys(self):
        long_record = _make_record(subject="Long one")
        long_record["stories"][0]["text"] = "A very long story that needs scrolling. " * 40
        app = ReviewApp([long_record])
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            scroll = app.screen.query_one("#detail-scroll")
            assert scroll.scroll_offset.y == 0
            await pilot.press("down", "down", "down")
            assert scroll.scroll_offset.y == 3

    async def test_enter_on_empty_filtered_list_is_noop(self):
        app = ReviewApp(_ui_records(), init_sender="nobody@nowhere")
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            assert not isinstance(app.screen, DetailScreen)


class TestReviewAppTierFilter:
    async def test_tier_filter_narrows_list(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "t", "g")
            assert len(app.query_one(ListView)) == 3
            assert "tier:good" in _title(app)
            assert "3/6 records" in _title(app)

    async def test_clear_tier_filter_restores_all(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "t", "g")
            assert len(app.query_one(ListView)) == 3
            await pilot.press("f", "t", "c")
            assert len(app.query_one(ListView)) == 6

    async def test_cancel_at_filter_type_menu(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "z")  # z is not a filter type -> cancel
            assert len(app.query_one(ListView)) == 6
            assert "tier:" not in _title(app)

    async def test_cancel_at_tier_menu_keeps_existing_filter(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "t", "g")
            await pilot.press("f", "t", "z")  # z is not a tier key -> cancel
            assert len(app.query_one(ListView)) == 3
            assert "tier:good" in _title(app)

    async def test_filter_change_resets_cursor(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("down", "down")
            await pilot.press("f", "t", "g")
            assert app.query_one(ListView).index == 0


class TestReviewAppThemeFilter:
    async def test_theme_filter_narrows_list(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "h", "v")  # filter -> theme -> vocation_family
            assert len(app.query_one(ListView)) == 1
            assert "theme:vocation_family" in _title(app)
            assert "1/6 records" in _title(app)

    async def test_clear_theme_filter_restores_all(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "h", "s")  # theme -> scripture (5 records)
            assert len(app.query_one(ListView)) == 5
            await pilot.press("f", "h", "x")  # x clears the theme filter
            assert len(app.query_one(ListView)) == 6

    async def test_cancel_at_theme_menu(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "h", "z")  # z is not a theme key -> cancel
            assert len(app.query_one(ListView)) == 6


class TestReviewAppSenderFilter:
    async def test_sender_filter_narrows_list(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "s")
            await pilot.press(*"dm.org")
            await pilot.press("enter")
            assert len(app.query_one(ListView)) == 4
            assert "sender:dm.org" in _title(app)
            assert "4/6 records" in _title(app)

    async def test_sender_escape_cancels(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "s")
            await pilot.press(*"dm.org")
            await pilot.press("escape")
            assert len(app.query_one(ListView)) == 6
            assert "sender:" not in _title(app)

    async def test_sender_empty_submit_clears_filter(self):
        app = ReviewApp(_ui_records(), init_sender="dm.org")
        async with app.run_test(size=SIZE) as pilot:
            assert len(app.query_one(ListView)) == 4
            await pilot.press("f", "s", "enter")
            assert len(app.query_one(ListView)) == 6


class TestReviewAppInitFilters:
    async def test_init_tier_pre_applied(self):
        app = ReviewApp(_ui_records(), init_tier="good")
        async with app.run_test(size=SIZE):
            assert len(app.query_one(ListView)) == 3
            assert "tier:good" in _title(app)

    async def test_init_filters_combine(self):
        app = ReviewApp(_ui_records(), init_tier="good", init_sender="dm.org")
        async with app.run_test(size=SIZE):
            assert len(app.query_one(ListView)) == 3
            assert "tier:good" in _title(app)
            assert "sender:dm.org" in _title(app)


class TestRunReviewTui:
    def test_empty_records_prints_and_returns(self, capsys):
        run_review_tui([])
        assert "No assessment records" in capsys.readouterr().out


class TestReviewAppRobustness:
    async def test_filter_menu_survives_key_auto_repeat(self):
        # Two back-to-back keys (terminal auto-repeat) must not double-dismiss
        # the modal and crash the app with ScreenStackError.
        from textual import events

        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f")
            app.post_message(events.Key("t", "t"))
            app.post_message(events.Key("t", "t"))
            await pilot.pause()
            assert app.is_running
            await pilot.press("g")
            assert "tier:good" in _title(app)

    async def test_filter_keys_are_case_insensitive(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "T", "G")  # uppercase, like the curses .lower()
            assert "tier:good" in _title(app)

    async def test_typing_cancel_as_sender_filter_is_a_filter_not_a_dismissal(self):
        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "s")
            await pilot.press(*"cancel")
            await pilot.press("enter")
            assert "sender:cancel" in _title(app)

    async def test_list_page_and_home_end_move_cursor(self):
        from textual.widgets import ListView

        records = [_make_record(subject=f"r{i}") for i in range(20)]
        app = ReviewApp(records)
        async with app.run_test(size=(100, 8)) as pilot:
            # title + header + help = 3 rows -> list height 5
            await pilot.press("pagedown")
            assert app.query_one(ListView).index == 5
            await pilot.press("end")
            assert app.query_one(ListView).index == 19
            await pilot.press("home")
            assert app.query_one(ListView).index == 0

    async def test_ctrl_b_and_ctrl_f_page_the_list_cursor(self):
        from textual.widgets import ListView

        records = [_make_record(subject=f"r{i}") for i in range(20)]
        app = ReviewApp(records)
        async with app.run_test(size=(100, 8)) as pilot:
            await pilot.press("ctrl+f")
            assert app.query_one(ListView).index == 5
            await pilot.press("ctrl+b")
            assert app.query_one(ListView).index == 0

    async def test_detail_rewraps_on_resize(self):
        words = [f"word{i:02d}" for i in range(24)]
        record = _make_record(subject="Long")
        record["stories"][0]["text"] = " ".join(words)
        app = ReviewApp([record])
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("enter")
            await pilot.resize_terminal(40, 30)
            await pilot.pause()
            text = "\n".join(str(w.render()) for w in app.screen.query(Static))
            for word in words:
                assert word in text


class TestReviewAppReviewFindings:
    async def test_sender_filter_survives_double_enter(self):
        from textual import events

        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "s")
            await pilot.press(*"dm.org")
            app.post_message(events.Key("enter", None))
            app.post_message(events.Key("enter", None))
            await pilot.pause()
            assert app.is_running
            assert "sender:dm.org" in _title(app)

    async def test_sender_filter_strips_control_characters(self):
        from textual.widgets import Input

        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "s")
            app.screen.query_one(Input).value = "dm\x1b[31m.org"
            await pilot.press("enter")
            assert app.f_sender is not None
            assert "\x1b" not in app.f_sender

    async def test_resize_preserves_list_cursor(self):
        records = [_make_record(subject=f"r{i}") for i in range(10)]
        app = ReviewApp(records)
        async with app.run_test(size=(100, 12)) as pilot:
            await pilot.press("down", "down", "down")
            assert app.query_one(ListView).index == 3
            await pilot.resize_terminal(120, 14)
            await pilot.pause()
            assert app.query_one(ListView).index == 3

    async def test_enter_auto_repeat_opens_a_single_detail(self):
        from textual import events

        app = ReviewApp(_ui_records())
        async with app.run_test(size=SIZE) as pilot:
            app.post_message(events.Key("enter", None))
            app.post_message(events.Key("enter", None))
            await pilot.pause()
            assert len(app.screen_stack) == 2  # base + ONE detail
            await pilot.press("escape")
            assert not isinstance(app.screen, DetailScreen)


def _dated_records():
    """Records with distinct fixed send-dates (all in 2024) for sort/date tests."""
    return [
        _make_record(subject="Jan", send_date="2024-01-10T00:00:00+00:00"),
        _make_record(subject="Mar", send_date="2024-03-10T00:00:00+00:00"),
        _make_record(subject="Feb", send_date="2024-02-10T00:00:00+00:00"),
    ]


class TestReviewAppSort:
    async def test_default_sort_by_send_date_desc(self):
        app = ReviewApp(_dated_records())
        async with app.run_test(size=SIZE):
            # Newest send-date first.
            assert [r["subject"] for r in app.filtered] == ["Mar", "Feb", "Jan"]
            first_row = str(app.query_one(ListView).children[0].query_one(Label).render())
            assert first_row.startswith("2024-03-10")


class TestReviewAppDateFilter:
    async def test_since_via_date_menu_narrows(self):
        app = ReviewApp(_dated_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "d", "s")            # filter -> date -> since…
            await pilot.press(*"2024-02-15")            # type the date
            await pilot.press("enter")
            assert "since:2024-02-15" in _title(app)
            # Only Mar (2024-03-10) is on/after the cutoff.
            assert [r["subject"] for r in app.filtered] == ["Mar"]

    async def test_past_30_days_excludes_old_fixed_dates(self):
        # The fixture's 2024 dates are far older than 30 days from "now", so a
        # "Past 30 days" filter empties the list — deterministic regardless of clock.
        app = ReviewApp(_dated_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "d", "3")
            assert "since:" in _title(app)
            assert len(app.query_one(ListView)) == 0

    async def test_clear_date_filter_restores_all(self):
        app = ReviewApp(_dated_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "d", "3")
            assert len(app.query_one(ListView)) == 0
            await pilot.press("f", "d", "x")            # clear
            assert len(app.query_one(ListView)) == 3
            assert "since:" not in _title(app)

    async def test_invalid_since_date_shows_hint_and_keeps_filter(self):
        app = ReviewApp(_dated_records())
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("f", "d", "s")
            await pilot.press(*"not-a-date")
            await pilot.press("enter")
            assert app.f_since is None            # rejected
            assert len(app.query_one(ListView)) == 3  # unfiltered
