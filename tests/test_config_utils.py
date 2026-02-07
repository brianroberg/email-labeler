"""Tests for config preprocessing utilities — {env.VAR_NAME} substitution."""

from config_utils import substitute_env_vars


class TestSubstituteString:
    def test_single_env_var(self, monkeypatch):
        monkeypatch.setenv("USER_NAME", "Alice")
        result = substitute_env_vars({"k": "Hello {env.USER_NAME}"})
        assert result["k"] == "Hello Alice"

    def test_multiple_env_vars(self, monkeypatch):
        monkeypatch.setenv("FIRST", "Alice")
        monkeypatch.setenv("SECOND", "Bob")
        result = substitute_env_vars({"k": "{env.FIRST} and {env.SECOND}"})
        assert result["k"] == "Alice and Bob"

    def test_preserves_runtime_placeholders(self, monkeypatch):
        monkeypatch.setenv("USER_NAME", "John")
        result = substitute_env_vars({"k": "Hello {env.USER_NAME}, from {sender}"})
        assert result["k"] == "Hello John, from {sender}"

    def test_missing_env_var_becomes_empty(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT", raising=False)
        result = substitute_env_vars({"k": "Hello {env.NONEXISTENT}"})
        assert result["k"] == "Hello "

    def test_no_env_vars_unchanged(self):
        result = substitute_env_vars({"k": "Hello {sender}"})
        assert result["k"] == "Hello {sender}"

    def test_empty_string(self):
        result = substitute_env_vars({"k": ""})
        assert result["k"] == ""

    def test_lowercase_var_name(self, monkeypatch):
        monkeypatch.setenv("lower_case", "works")
        result = substitute_env_vars({"k": "{env.lower_case}"})
        assert result["k"] == "works"


class TestRecursiveWalk:
    def test_nested_dicts(self, monkeypatch):
        monkeypatch.setenv("X", "val")
        config = {"a": {"b": {"c": "{env.X}"}}}
        assert substitute_env_vars(config)["a"]["b"]["c"] == "val"

    def test_lists(self, monkeypatch):
        monkeypatch.setenv("X", "val")
        config = {"items": ["{env.X}", "static", "{sender}"]}
        result = substitute_env_vars(config)
        assert result["items"] == ["val", "static", "{sender}"]

    def test_non_string_types_pass_through(self):
        config = {"n": 42, "f": 3.14, "b": True, "none": None}
        assert substitute_env_vars(config) == config

    def test_empty_dict(self):
        assert substitute_env_vars({}) == {}


class TestRealisticConfig:
    def test_email_classification_prompt(self, monkeypatch):
        """Simulates the actual config.toml pattern."""
        monkeypatch.setenv("USER_NAME", "John")
        config = {
            "prompts": {
                "email_classification": {
                    "categories": {
                        "NEEDS_RESPONSE": (
                            "Requires a reply from {env.USER_NAME}."
                            " Questions directed at {env.USER_NAME}."
                        ),
                        "FYI": "Informational, no action needed.",
                    },
                    "user_template": "From: {sender}\nSubject: {subject}\nBody:\n{body}",
                }
            }
        }
        result = substitute_env_vars(config)
        cats = result["prompts"]["email_classification"]["categories"]
        assert cats["NEEDS_RESPONSE"] == (
            "Requires a reply from John. Questions directed at John."
        )
        assert cats["FYI"] == "Informational, no action needed."
        # Runtime placeholders preserved
        tmpl = result["prompts"]["email_classification"]["user_template"]
        assert "{sender}" in tmpl
        assert "{subject}" in tmpl
        assert "{body}" in tmpl

    def test_idempotent(self, monkeypatch):
        monkeypatch.setenv("VAR", "value")
        config = {"k": "{env.VAR}"}
        once = substitute_env_vars(config)
        twice = substitute_env_vars(once)
        assert once == twice == {"k": "value"}
