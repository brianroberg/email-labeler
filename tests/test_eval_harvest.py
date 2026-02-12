"""Tests for evals.harvest — ground truth inference and deduplication."""

import json

from evals.harvest import deduplicate, infer_ground_truth
from evals.schemas import GoldenThread

# Label config matching the real config.toml structure
LABELS_CONFIG = {
    "needs_response": "agent/needs-response",
    "fyi": "agent/fyi",
    "low_priority": "agent/low-priority",
    "processed": "agent/processed",
    "personal": "agent/personal",
    "non_personal": "agent/non-personal",
}

# Label ID -> name mapping (as returned by Gmail API list_labels)
LABEL_ID_TO_NAME = {
    "Label_1": "agent/needs-response",
    "Label_2": "agent/fyi",
    "Label_3": "agent/low-priority",
    "Label_4": "agent/processed",
    "Label_5": "agent/personal",
    "Label_6": "agent/non-personal",
}


class TestInferGroundTruth:
    def test_person_needs_response(self):
        messages = [
            {"labelIds": ["INBOX", "Label_5", "Label_1", "Label_4"]},
        ]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == "person"
        assert label == "needs_response"

    def test_service_low_priority(self):
        messages = [
            {"labelIds": ["Label_6", "Label_3", "Label_4"]},
        ]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == "service"
        assert label == "low_priority"

    def test_person_fyi(self):
        messages = [
            {"labelIds": ["Label_5", "Label_2", "Label_4"]},
        ]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == "person"
        assert label == "fyi"

    def test_multi_message_thread(self):
        """Labels on any message in the thread should be picked up."""
        messages = [
            {"labelIds": ["INBOX"]},
            {"labelIds": ["Label_5"]},  # personal on second message
            {"labelIds": ["Label_1", "Label_4"]},  # needs_response on third
        ]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == "person"
        assert label == "needs_response"

    def test_missing_sender_type(self):
        """Missing sender type label should return empty string."""
        messages = [
            {"labelIds": ["Label_1", "Label_4"]},  # Has label but no personal/non_personal
        ]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == ""
        assert label == "needs_response"

    def test_missing_classification_label(self):
        """Missing classification label should return empty string."""
        messages = [
            {"labelIds": ["Label_5", "Label_4"]},  # Has personal but no classification
        ]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == "person"
        assert label == ""

    def test_no_labels(self):
        messages = [{"labelIds": []}]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == ""
        assert label == ""

    def test_unknown_label_ids(self):
        messages = [{"labelIds": ["UNKNOWN_1", "UNKNOWN_2"]}]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == ""
        assert label == ""

    def test_missing_labelIds_key(self):
        """Messages without labelIds should be handled gracefully."""
        messages = [{}]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == ""
        assert label == ""


class TestDeduplicate:
    def _make_golden(self, thread_id: str) -> GoldenThread:
        return GoldenThread(
            thread_id=thread_id,
            messages=[],
            senders=["test@example.com"],
            subject="Test",
            snippet="Test",
            expected_sender_type="service",
            expected_label="low_priority",
        )

    def test_no_existing_file(self, tmp_path):
        """With no existing file, all threads should be kept."""
        threads = [self._make_golden("t1"), self._make_golden("t2")]
        result = deduplicate(threads, tmp_path / "nonexistent.jsonl")
        assert len(result) == 2

    def test_dedup_removes_existing(self, tmp_path):
        """Threads already in the file should be removed."""
        existing_file = tmp_path / "golden.jsonl"
        existing_file.write_text(
            json.dumps({"thread_id": "t1", "messages": [], "senders": [], "subject": "",
                        "snippet": "", "expected_sender_type": "service",
                        "expected_label": "low_priority"}) + "\n"
        )
        threads = [self._make_golden("t1"), self._make_golden("t2"), self._make_golden("t3")]
        result = deduplicate(threads, existing_file)
        assert len(result) == 2
        assert {t.thread_id for t in result} == {"t2", "t3"}

    def test_all_duplicates(self, tmp_path):
        """If all threads are duplicates, result should be empty."""
        existing_file = tmp_path / "golden.jsonl"
        lines = [
            json.dumps({"thread_id": "t1"}) + "\n",
            json.dumps({"thread_id": "t2"}) + "\n",
        ]
        existing_file.write_text("".join(lines))
        threads = [self._make_golden("t1"), self._make_golden("t2")]
        result = deduplicate(threads, existing_file)
        assert len(result) == 0

    def test_empty_new_threads(self, tmp_path):
        result = deduplicate([], tmp_path / "golden.jsonl")
        assert len(result) == 0
