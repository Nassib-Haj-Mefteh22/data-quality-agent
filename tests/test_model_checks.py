"""Tests for the check functions of scripts/test_models.py. They make no network call."""

from test_models import check_report, check_tool_call

EXPECTED = {"file": "sales.csv"}


def test_correct_tool_call_passes() -> None:
    result = check_tool_call("count_duplicates", '{"file": "sales.csv"}', "count_duplicates", EXPECTED)
    assert result == "pass"


def test_wrong_tool_is_detected() -> None:
    result = check_tool_call("column_stats", '{"file": "sales.csv"}', "count_duplicates", EXPECTED)
    assert result == "wrong tool"


def test_invalid_json_is_detected() -> None:
    result = check_tool_call("count_duplicates", '{"file": ', "count_duplicates", EXPECTED)
    assert result == "invalid JSON"


def test_wrong_arguments_are_detected() -> None:
    result = check_tool_call("count_duplicates", '{"file": "other.csv"}', "count_duplicates", EXPECTED)
    assert result == "wrong arguments"


def test_correct_report_passes() -> None:
    text = (
        '{"issue_type": "impossible_value", "column": "age", '
        '"rows": [3, 7], "evidence": "Age cannot be negative."}'
    )
    assert check_report(text) == "pass"


def test_missing_key_is_a_bad_format() -> None:
    assert check_report('{"column": "age"}') == "bad format"


def test_wrong_rows_are_detected() -> None:
    text = (
        '{"issue_type": "impossible_value", "column": "age", '
        '"rows": [3], "evidence": "Age cannot be negative."}'
    )
    assert check_report(text) == "wrong rows"
