"""Shared fixtures and sample Gmail data for tests."""

import base64

import pytest


@pytest.fixture
def sample_headers():
    """Sample Gmail message headers."""
    return [
        {"name": "From", "value": "John Doe <john@example.com>"},
        {"name": "To", "value": "me@example.com"},
        {"name": "Subject", "value": "Meeting tomorrow"},
        {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
    ]


@pytest.fixture
def sample_message():
    """Sample full Gmail message resource."""
    body_text = "Hey, can we meet tomorrow at 3pm to discuss the project?"
    encoded_body = base64.urlsafe_b64encode(body_text.encode()).decode()
    return {
        "id": "msg_001",
        "threadId": "thread_001",
        "labelIds": ["INBOX", "UNREAD"],
        "snippet": "Hey, can we meet tomorrow at 3pm to discuss the project?",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "John Doe <john@example.com>"},
                {"name": "To", "value": "me@example.com"},
                {"name": "Subject", "value": "Meeting tomorrow"},
                {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
            ],
            "body": {"data": encoded_body, "size": len(body_text)},
        },
    }


@pytest.fixture
def sample_service_message():
    """Sample service/automated Gmail message resource."""
    body_text = "Your order #12345 has shipped! Track your package at https://example.com/track"
    encoded_body = base64.urlsafe_b64encode(body_text.encode()).decode()
    return {
        "id": "msg_002",
        "threadId": "thread_002",
        "labelIds": ["INBOX", "UNREAD"],
        "snippet": "Your order #12345 has shipped!",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "Amazon <shipment-tracking@amazon.com>"},
                {"name": "To", "value": "me@example.com"},
                {"name": "Subject", "value": "Your Amazon order has shipped"},
                {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
            ],
            "body": {"data": encoded_body, "size": len(body_text)},
        },
    }


@pytest.fixture
def sample_labels():
    """Sample Gmail labels list response."""
    return {
        "labels": [
            {"id": "INBOX", "name": "INBOX", "type": "system"},
            {"id": "Label_1", "name": "agent/needs-response", "type": "user"},
            {"id": "Label_2", "name": "agent/fyi", "type": "user"},
            {"id": "Label_3", "name": "agent/low-priority", "type": "user"},
            {"id": "Label_4", "name": "agent/unwanted", "type": "user"},
            {"id": "Label_5", "name": "agent/processed", "type": "user"},
            {"id": "Label_6", "name": "agent/would-have-deleted", "type": "user"},
        ]
    }
