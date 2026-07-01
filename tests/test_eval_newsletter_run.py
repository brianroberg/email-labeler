"""Tests for the newsletter eval runner."""

import argparse
import asyncio
import copy
import json
from pathlib import Path

import pytest

from evals import newsletter_run
from evals.llm_cache import CachedLLMClient
from evals.newsletter_run import (
    build_meta,
    build_output_path,
    compute_prompt_hash,
    evaluate_extraction,
    evaluate_story,
    load_golden_set,
    merge_prompts_override,
    run_evaluation,
    write_results,
    write_thinking_sidecar,
)
from evals.newsletter_schemas import GoldenNewsletter, GoldenStory
from newsletter import NewsletterClassifier


class FakeLLM:
    """Fake inner LLMClient: returns canned (response, thinking) per prompt.

    Keyed by a substring match on the user_content so a single instance can
    answer extraction / quality / theme calls differently. Records every call.
    """

    def __init__(self, responses):
        # responses: list of (needle, response, thinking); first needle-in-user wins
        self.responses = responses
        self.model = "fake-model"
        self.temperature = 0.0
        self.max_tokens = 100
        self.extra_body = {}
        self.calls = []

    async def complete(self, system_prompt, user_content, include_thinking=False):
        self.calls.append((system_prompt, user_content))
        for needle, response, thinking in self.responses:
            if needle in user_content:
                return (response, thinking) if include_thinking else response
        return ("", "") if include_thinking else ""

    async def is_available(self, timeout=None):
        return True


def _classifier(llm):
    config = _prompts_config()
    return NewsletterClassifier(cloud_llm=llm, config=config)


def _run(coro):
    return asyncio.run(coro)


def _newsletter(thread_id, stories=None, **kw):
    base = dict(
        thread_id=thread_id, message_id=f"m-{thread_id}", sender="s@dm.org",
        subject="subj", body="body text", stories=stories or [],
    )
    base.update(kw)
    return GoldenNewsletter(**base)


def _story(story_id, **kw):
    base = dict(story_id=story_id, title="t", text="x")
    base.update(kw)
    return GoldenStory(**base)


def _write_golden(path, newsletters):
    path.write_text("".join(json.dumps(n.to_dict()) + "\n" for n in newsletters))


def _prompts_config():
    return {
        "newsletter": {
            "prompts": {
                "story_extraction": {"system": "extract sys", "user_template": "EXTRACT {body}"},
                "quality_assessment": {"system": "quality sys", "user_template": "QUALITY {title} {text}"},
                "theme_classification": {"system": "theme sys", "user_template": "THEME {title} {text}"},
            }
        }
    }


class TestComputePromptHash:
    def test_stable_for_identical_config(self):
        a = compute_prompt_hash(_prompts_config())
        b = compute_prompt_hash(_prompts_config())
        assert a == b
        assert len(a) == 16

    def test_changes_when_a_prompt_string_changes(self):
        base = _prompts_config()
        mutated = copy.deepcopy(base)
        mutated["newsletter"]["prompts"]["quality_assessment"]["system"] = "different"
        assert compute_prompt_hash(base) != compute_prompt_hash(mutated)


class TestMergePromptsOverride:
    def test_deep_merges_and_changes_prompt_hash(self):
        base = _prompts_config()
        before = compute_prompt_hash(base)
        override = {
            "newsletter": {
                "prompts": {"quality_assessment": {"system": "NEW quality sys"}}
            }
        }
        merge_prompts_override(base, override)
        # overridden key changed
        assert base["newsletter"]["prompts"]["quality_assessment"]["system"] == "NEW quality sys"
        # sibling keys in the same block preserved (deep merge, not replace)
        qa = base["newsletter"]["prompts"]["quality_assessment"]
        assert qa["user_template"] == "QUALITY {title} {text}"
        # other prompt blocks untouched
        assert base["newsletter"]["prompts"]["story_extraction"]["system"] == "extract sys"
        # and the hash moved
        assert compute_prompt_hash(base) != before

    def test_only_touches_newsletter_prompts(self):
        base = _prompts_config()
        base["newsletter"]["llm"] = {"model": "keep-me"}
        override = {"newsletter": {"llm": {"model": "IGNORED"}, "prompts": {}}}
        merge_prompts_override(base, override)
        assert base["newsletter"]["llm"]["model"] == "keep-me"


class TestLoadGoldenSet:
    def test_reviewed_only_filters_unreviewed(self, tmp_path):
        path = tmp_path / "g.jsonl"
        _write_golden(path, [
            _newsletter("keep", reviewed=True),
            _newsletter("drop", reviewed=False),
        ])
        loaded = load_golden_set(path, reviewed_only=True)
        assert [n.thread_id for n in loaded] == ["keep"]

    def test_excluded_dropped_even_when_unreviewed_included(self, tmp_path):
        path = tmp_path / "g.jsonl"
        _write_golden(path, [
            _newsletter("keep", reviewed=False),
            _newsletter("drop", reviewed=False, excluded=True),
        ])
        loaded = load_golden_set(path, reviewed_only=False)
        assert [n.thread_id for n in loaded] == ["keep"]

    def test_tolerates_blank_lines(self, tmp_path):
        path = tmp_path / "g.jsonl"
        content = json.dumps(_newsletter("keep", reviewed=True).to_dict()) + "\n\n"
        path.write_text(content)
        loaded = load_golden_set(path, reviewed_only=True)
        assert [n.thread_id for n in loaded] == ["keep"]


class TestEvaluateExtraction:
    def test_records_predicted_and_golden_story_sets(self):
        extraction_out = "TITLE: Pred One\nTEXT: pred body one\n\nTITLE: Pred Two\nTEXT: pred body two"
        llm = FakeLLM([("EXTRACT", extraction_out, "")])
        newsletter = _newsletter(
            "nl1", reviewed=True,
            stories=[
                _story("nl1:0", title="Gold One", text="gold body one"),
                _story("nl1:1", title="Gold Two", text="gold body two"),
            ],
        )
        pred = _run(evaluate_extraction(newsletter, _classifier(llm)))

        assert pred.thread_id == "nl1"
        assert pred.predicted_stories == [
            {"title": "Pred One", "text": "pred body one"},
            {"title": "Pred Two", "text": "pred body two"},
        ]
        assert pred.golden_stories == [
            {"story_id": "nl1:0", "title": "Gold One", "text": "gold body one"},
            {"story_id": "nl1:1", "title": "Gold Two", "text": "gold body two"},
        ]
        assert pred.error is None


class TestEvaluateStory:
    def _story_llm(self):
        quality_out = "SIMPLE: 4\nCONCRETE: 5\nPERSONAL: 4\nDYNAMIC: 3"
        theme_out = "SCRIPTURE\nCHURCH"
        return FakeLLM([
            ("QUALITY", quality_out, "quality reasoning"),
            ("THEME", theme_out, "theme reasoning"),
        ])

    def test_records_scores_themes_and_derives_tier(self):
        llm = self._story_llm()
        story = _story(
            "nl1:0", title="A Story", text="the text",
            expected_scores={"simple": 4, "concrete": 4, "personal": 4, "dynamic": 4},
            expected_tier="excellent", expected_themes=["scripture"],
        )
        pred, thinking = _run(evaluate_story(story, "nl1", _classifier(llm)))

        assert pred.story_id == "nl1:0"
        assert pred.thread_id == "nl1"
        # expected_* carried from the golden story
        assert pred.expected_scores == {"simple": 4, "concrete": 4, "personal": 4, "dynamic": 4}
        assert pred.expected_tier == "excellent"
        assert pred.expected_themes == ["scripture"]
        # predicted_*
        assert pred.predicted_scores == {"simple": 4, "concrete": 5, "personal": 4, "dynamic": 3}
        assert pred.predicted_themes == ["scripture", "church"]
        # tier derived via compute_tier: avg = (4+5+4+3)/4 = 4.0 -> excellent
        assert pred.predicted_tier == "excellent"
        # raw captured
        assert "SIMPLE: 4" in pred.scores_raw
        assert "SCRIPTURE" in pred.themes_raw
        assert pred.error is None
        # thinking sidecar carries both cots
        assert thinking.story_id == "nl1:0"
        assert thinking.quality_cot == "quality reasoning"
        assert thinking.theme_cot == "theme reasoning"

    def test_predicted_tier_none_when_scores_unparseable(self):
        llm = FakeLLM([
            ("QUALITY", "garbage no scores", "q"),
            ("THEME", "NONE", "t"),
        ])
        story = _story("nl2:0", title="T", text="x")
        pred, _thinking = _run(evaluate_story(story, "nl2", _classifier(llm)))
        assert pred.predicted_scores is None
        assert pred.predicted_tier is None
        assert pred.predicted_themes == []


def _full_llm():
    return FakeLLM([
        ("EXTRACT", "TITLE: S\nTEXT: body", "ecot"),
        ("QUALITY", "SIMPLE: 3\nCONCRETE: 3\nPERSONAL: 3\nDYNAMIC: 3", "qcot"),
        ("THEME", "SCRIPTURE", "tcot"),
    ])


class TestRunEvaluation:
    def _golden(self):
        return [_newsletter(
            "nl1", reviewed=True,
            stories=[_story("nl1:0", title="S", text="body")],
        )]

    def test_all_mode_produces_extraction_and_story_rows(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        cached = CachedLLMClient(_full_llm(), cache)
        rows, thinking = _run(run_evaluation(
            self._golden(), _classifier(cached), mode="all", parallelism=1,
        ))
        from evals.newsletter_schemas import ExtractionPrediction, StoryPrediction
        extractions = [r for r in rows if isinstance(r, ExtractionPrediction)]
        stories = [r for r in rows if isinstance(r, StoryPrediction)]
        assert len(extractions) == 1
        assert len(stories) == 1
        assert stories[0].predicted_tier == "good"  # avg 3.0
        assert thinking[0].quality_cot == "qcot"

    def test_second_identical_run_records_cache_hits(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        cached = CachedLLMClient(_full_llm(), cache)
        _run(run_evaluation(self._golden(), _classifier(cached), mode="all", parallelism=1))
        first_misses = cached.misses
        assert first_misses > 0
        cached.flush()

        # Fresh cache client over the same on-disk cache: every call is now a hit.
        cached2 = CachedLLMClient(_full_llm(), cache)
        _run(run_evaluation(self._golden(), _classifier(cached2), mode="all", parallelism=1))
        assert cached2.hits == first_misses
        assert cached2.misses == 0

    def test_extraction_mode_skips_story_scoring(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        cached = CachedLLMClient(_full_llm(), cache)
        rows, _thinking = _run(run_evaluation(
            self._golden(), _classifier(cached), mode="extraction", parallelism=1,
        ))
        from evals.newsletter_schemas import ExtractionPrediction, StoryPrediction
        assert all(isinstance(r, ExtractionPrediction) for r in rows)
        assert not any(isinstance(r, StoryPrediction) for r in rows)

    def test_quality_mode_skips_extraction(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        cached = CachedLLMClient(_full_llm(), cache)
        rows, _thinking = _run(run_evaluation(
            self._golden(), _classifier(cached), mode="quality", parallelism=1,
        ))
        from evals.newsletter_schemas import ExtractionPrediction, StoryPrediction
        assert all(isinstance(r, StoryPrediction) for r in rows)
        assert not any(isinstance(r, ExtractionPrediction) for r in rows)

    def test_excluded_story_skipped_in_quality_mode(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        cached = CachedLLMClient(_full_llm(), cache)
        golden = [_newsletter(
            "nl1", reviewed=True,
            stories=[
                _story("nl1:0", title="S", text="body"),
                _story("nl1:1", title="X", text="y", excluded=True),
            ],
        )]
        rows, _thinking = _run(run_evaluation(
            golden, _classifier(cached), mode="quality", parallelism=1,
        ))
        assert [r.story_id for r in rows] == ["nl1:0"]


def _full_config():
    cfg = _prompts_config()
    cfg["newsletter"]["llm"] = {
        "model": "claude-x", "temperature": 0.0, "max_tokens": 1024, "extra_body": None,
    }
    return cfg


class TestBuildMeta:
    def test_records_prompt_hash_model_and_system_prompts(self):
        cfg = _full_config()
        golden = [_newsletter("nl1", reviewed=True, seeded_from="parse_stories",
                              stories=[_story("nl1:0"), _story("nl1:1")])]
        meta = build_meta(
            config=cfg, config_path="config.toml", golden_path="g.jsonl",
            golden_set=golden, mode="all", tag="baseline", parallelism=2,
        )
        assert meta.prompt_hash == compute_prompt_hash(cfg)
        assert meta.newsletter_model == "claude-x"
        assert meta.mode == "all"
        assert meta.tag == "baseline"
        assert meta.golden_set_count == 1
        assert meta.story_count == 2
        assert meta.parallelism == 2
        assert meta.extraction_system_prompt == "extract sys"
        assert meta.quality_system_prompt == "quality sys"
        assert meta.theme_system_prompt == "theme sys"
        assert meta.temperature == 0.0
        assert meta.max_tokens == 1024
        # seeded_from pulled from the golden set
        assert meta.seeded_from == "parse_stories"


class TestBuildOutputPath:
    def test_filename_has_mode_tag_and_runid(self):
        p = build_output_path(Path("/out"), "all", "baseline", "abcdef1234567890")
        assert p.parent == Path("/out")
        assert p.name.endswith(".jsonl")
        assert "_all_" in p.name
        assert "baseline" in p.name
        assert "abcdef12" in p.name  # first 8 of run id

    def test_omits_empty_tag(self):
        p = build_output_path(Path("/out"), "quality", "", "abcdef1234567890")
        assert "quality" in p.name
        assert "__" not in p.name.replace(".jsonl", "")


class TestWriteResults:
    def test_meta_first_then_rows_roundtrip(self, tmp_path):
        from evals.newsletter_schemas import (
            ExtractionPrediction,
            NewsletterRunMeta,
            StoryPrediction,
        )
        meta = NewsletterRunMeta(
            run_id="r", timestamp="t", config_hash="c", config_path="p",
            newsletter_model="m", golden_set_path="g", golden_set_count=1, story_count=1,
        )
        rows = [
            ExtractionPrediction(thread_id="nl1"),
            StoryPrediction(story_id="nl1:0", thread_id="nl1"),
        ]
        path = tmp_path / "out.jsonl"
        write_results(path, meta, rows)
        lines = path.read_text().splitlines()
        assert json.loads(lines[0])["type"] == "run_meta"
        assert json.loads(lines[1])["type"] == "extraction_prediction"
        assert json.loads(lines[2])["type"] == "story_prediction"


class TestWriteThinkingSidecar:
    def test_writes_only_nonempty_entries(self, tmp_path):
        from evals.newsletter_schemas import NewsletterThinkingEntry
        path = tmp_path / "out.jsonl"
        entries = [
            NewsletterThinkingEntry(story_id="a", quality_cot="q"),
            NewsletterThinkingEntry(story_id="b"),  # empty -> skipped
        ]
        write_thinking_sidecar(path, entries)
        sidecar = path.with_suffix(".cot.jsonl")
        lines = sidecar.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["story_id"] == "a"

    def test_no_sidecar_when_all_empty(self, tmp_path):
        from evals.newsletter_schemas import NewsletterThinkingEntry
        path = tmp_path / "out.jsonl"
        write_thinking_sidecar(path, [NewsletterThinkingEntry(story_id="a")])
        assert not path.with_suffix(".cot.jsonl").exists()


def _main_args(**kw):
    base = dict(
        golden_set="", config=None, output_dir="", mode="all", tag=None,
        no_cache=True, parallelism=None, include_unreviewed=False,
        prompts=None, model=None, report=False, compare_to=None, skip_preflight=True,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _read_meta(output_dir):
    files = sorted(Path(output_dir).glob("*.jsonl"))
    files = [f for f in files if not f.name.endswith(".cot.jsonl")]
    assert files, "no results file written"
    return json.loads(files[-1].read_text().splitlines()[0])


class TestMainPromptsOverride:
    def _patch_classifier(self, monkeypatch):
        """Route main through a FakeLLM-backed classifier so no network is hit."""
        def fake_build(config, no_cache):
            llm = _full_llm()
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)

    def test_prompts_override_changes_recorded_prompt_hash(
        self, tmp_path, monkeypatch
    ):
        self._patch_classifier(monkeypatch)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True, stories=[_story("nl1:0", title="S", text="body")],
        )])
        out = tmp_path / "results"

        # Baseline run over the real config.toml
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out), mode="quality", tag="base",
        )))
        base_hash = _read_meta(out)["prompt_hash"]

        # Variant run with a --prompts override that changes the quality system prompt
        alt = tmp_path / "alt.toml"
        alt.write_text(
            '[newsletter.prompts.quality_assessment]\n'
            'system = "A COMPLETELY DIFFERENT QUALITY PROMPT"\n'
        )
        out2 = tmp_path / "results2"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out2), mode="quality",
            tag="variant", prompts=str(alt),
        )))
        variant_meta = _read_meta(out2)
        assert variant_meta["prompt_hash"] != base_hash
        assert variant_meta["quality_system_prompt"] == "A COMPLETELY DIFFERENT QUALITY PROMPT"

    def test_model_override_recorded(self, tmp_path, monkeypatch):
        self._patch_classifier(monkeypatch)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True, stories=[_story("nl1:0", title="S", text="body")],
        )])
        out = tmp_path / "results"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out), mode="extraction",
            model="my-override-model",
        )))
        assert _read_meta(out)["newsletter_model"] == "my-override-model"

    def test_no_newsletters_exits_1(self, tmp_path, monkeypatch):
        self._patch_classifier(monkeypatch)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter("nl1", reviewed=False)])  # unreviewed only
        with pytest.raises(SystemExit) as excinfo:
            asyncio.run(newsletter_run.main(_main_args(
                golden_set=str(golden), output_dir=str(tmp_path / "r"),
            )))
        assert excinfo.value.code == 1


class TestMainPreflight:
    def test_unreachable_endpoint_exits_1(self, tmp_path, monkeypatch):
        class DeadLLM(FakeLLM):
            async def is_available(self, timeout=None):
                return False

        def fake_build(config, no_cache):
            llm = DeadLLM([])
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)

        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True, stories=[_story("nl1:0", title="S", text="body")],
        )])
        args = _main_args(golden_set=str(golden), output_dir=str(tmp_path / "r"),
                          skip_preflight=False)
        with pytest.raises(SystemExit) as excinfo:
            asyncio.run(newsletter_run.main(args))
        assert excinfo.value.code == 1

    def test_skip_preflight_bypasses_check(self, tmp_path, monkeypatch):
        class DeadLLM(FakeLLM):
            async def is_available(self, timeout=None):
                return False

        def fake_build(config, no_cache):
            llm = DeadLLM([
                ("EXTRACT", "NO_STORIES", ""),
                ("QUALITY", "SIMPLE: 3\nCONCRETE: 3\nPERSONAL: 3\nDYNAMIC: 3", ""),
                ("THEME", "NONE", ""),
            ])
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)

        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True, stories=[_story("nl1:0", title="S", text="body")],
        )])
        out = tmp_path / "r"
        # Endpoint "down" but --skip-preflight -> main runs to completion and writes.
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out), skip_preflight=True,
        )))
        assert _read_meta(out)["type"] == "run_meta"


class TestMainReport:
    """--report / --compare-to must invoke the real newsletter_report API.

    Guards against the maybe_report() shim drifting from newsletter_report
    (which exposes compute_all_metrics + a 3-tuple load_results, not the
    email report's compute_metrics + 2-tuple load_results).
    """

    def _patch_classifier(self, monkeypatch):
        def fake_build(config, no_cache):
            llm = _full_llm()
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)

    def test_report_flag_prints_metrics(self, tmp_path, monkeypatch, capsys):
        self._patch_classifier(monkeypatch)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True,
            stories=[_story(
                "nl1:0", title="S", text="body",
                expected_scores={"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
                expected_tier="good", expected_themes=["scripture"],
            )],
        )])
        out = tmp_path / "r"
        # Must not raise (AttributeError on compute_metrics, tuple-unpack, etc.).
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out), mode="all", report=True,
        )))
        printed = capsys.readouterr().out
        assert "Newsletter Evaluation Report" in printed

    def test_compare_to_prints_comparison(self, tmp_path, monkeypatch, capsys):
        self._patch_classifier(monkeypatch)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True,
            stories=[_story(
                "nl1:0", title="S", text="body",
                expected_scores={"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
                expected_tier="good", expected_themes=["scripture"],
            )],
        )])
        # First run to produce a prior results file to compare against.
        out1 = tmp_path / "r1"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out1), mode="all",
        )))
        prior = sorted(
            f for f in Path(out1).glob("*.jsonl") if not f.name.endswith(".cot.jsonl")
        )[-1]

        out2 = tmp_path / "r2"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out2), mode="all",
            compare_to=str(prior),
        )))
        printed = capsys.readouterr().out
        assert "Newsletter Comparison Report" in printed
