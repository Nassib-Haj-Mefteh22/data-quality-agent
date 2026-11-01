"""Tests for config.py. They use a fake key and never read the real .env file."""

import pytest

from qualite import config


@pytest.fixture(autouse=True)
def ignore_real_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace load_dotenv, so each test controls the environment itself."""
    monkeypatch.setattr(config, "load_dotenv", lambda *args: False)


def test_missing_key_gives_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY is missing"):
        config.load_settings()


def test_blank_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "   ")

    with pytest.raises(RuntimeError, match="GROQ_API_KEY is missing"):
        config.load_settings()


def test_key_is_loaded_but_not_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    settings = config.load_settings()

    assert settings.groq_api_key == "fake-test-key"
    assert "fake-test-key" not in repr(settings)


def test_model_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")

    assert config.load_settings().model == "test-model"
