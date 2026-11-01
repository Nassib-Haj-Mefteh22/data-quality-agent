"""Quick filter: which Groq model does the three jobs our agent needs?

WHY this script: the docs say a model supports tools, not how well it uses them on our task.
WHY so small: this only removes bad candidates. The real comparison comes in Step 9.
WHY no cache yet: the cache arrives in feature 0.5, so this script makes few calls.
"""

import json
import time
from dataclasses import dataclass

from openai import APIError, OpenAI
from pydantic import BaseModel, ValidationError

from qualite.config import load_settings

MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]
PAUSE_SECONDS = 2.0

SYSTEM_PROMPT = "You audit data files. Use a tool when a tool can answer the question."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "count_duplicates",
            "description": "Count the exact duplicate rows in a CSV file.",
            "parameters": {
                "type": "object",
                "properties": {"file": {"type": "string"}},
                "required": ["file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "column_stats",
            "description": "Return the minimum, maximum and mean of one numeric column.",
            "parameters": {
                "type": "object",
                "properties": {"file": {"type": "string"}, "column": {"type": "string"}},
                "required": ["file", "column"],
            },
        },
    },
]

TOOL_CASES = [
    {
        "question": "How many duplicate rows are in sales.csv?",
        "tool": "count_duplicates",
        "arguments": {"file": "sales.csv"},
    },
    {
        "question": "What are the minimum and maximum of the price column in sales.csv?",
        "tool": "column_stats",
        "arguments": {"file": "sales.csv", "column": "price"},
    },
]

REPORT_PROMPT = """A tool found that column "age" has the value -5 at rows 3 and 7.
Return one JSON object with exactly these keys:
- issue_type: use the value "impossible_value"
- column: the column name
- rows: a list of row numbers
- evidence: one short sentence"""


class Issue(BaseModel):
    """The report format that the agent must be able to produce."""

    issue_type: str
    column: str
    rows: list[int]
    evidence: str


@dataclass
class CaseResult:
    """The outcome of one model on one test."""

    verdict: str
    seconds: float = 0.0
    tokens: int = 0


def check_tool_call(
    name: str, raw_arguments: str, expected_name: str, expected_arguments: dict[str, str]
) -> str:
    """Compare a requested tool call with what we expected.

    WHY separate: this has no network call, so pytest can test it directly.
    """
    if name != expected_name:
        return "wrong tool"
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return "invalid JSON"
    if arguments != expected_arguments:
        return "wrong arguments"
    return "pass"


def check_report(text: str) -> str:
    """Check the report format and its content, because valid JSON can still be wrong."""
    try:
        issue = Issue.model_validate_json(text)
    except ValidationError:
        return "bad format"
    if issue.issue_type != "impossible_value" or issue.column != "age":
        return "wrong content"
    if sorted(issue.rows) != [3, 7]:
        return "wrong rows"
    return "pass"


def run_tool_case(client: OpenAI, model: str, case: dict) -> CaseResult:
    """Ask one question and check the tool call, without running any tool."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": case["question"]},
    ]
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0
        )
    except APIError as error:
        return CaseResult(verdict=f"API error {getattr(error, 'status_code', '?')}")
    seconds = time.perf_counter() - started

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        verdict = "no tool call"
    else:
        call = tool_calls[0].function
        print(f"    asked for: {call.name}({call.arguments})")
        verdict = check_tool_call(call.name, call.arguments, case["tool"], case["arguments"])
    return CaseResult(verdict, seconds, response.usage.total_tokens)


def run_report_case(client: OpenAI, model: str) -> CaseResult:
    """Ask for a structured report and check it."""
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": REPORT_PROMPT}],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except APIError as error:
        return CaseResult(verdict=f"API error {getattr(error, 'status_code', '?')}")
    seconds = time.perf_counter() - started
    text = response.choices[0].message.content or ""
    return CaseResult(check_report(text), seconds, response.usage.total_tokens)


def test_model(client: OpenAI, model: str) -> list[CaseResult]:
    """Run the three tests on one model, pausing to respect the free limits."""
    print(f"\n{model}")
    results = []
    for case in TOOL_CASES:
        results.append(run_tool_case(client, model, case))
        time.sleep(PAUSE_SECONDS)
    results.append(run_report_case(client, model))
    time.sleep(PAUSE_SECONDS)
    return results


def print_summary(all_results: dict[str, list[CaseResult]]) -> None:
    """Print one line per model."""
    print(f"\n{'model':<28}{'tool 1':<18}{'tool 2':<18}{'report':<18}{'avg s':<8}{'tokens'}")
    for model, results in all_results.items():
        verdicts = "".join(f"{result.verdict:<18}" for result in results)
        timed = [result.seconds for result in results if result.seconds > 0]
        average = sum(timed) / len(timed) if timed else 0.0
        tokens = sum(result.tokens for result in results)
        print(f"{model:<28}{verdicts}{average:<8.2f}{tokens}")


def main() -> None:
    settings = load_settings()
    client = OpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
        max_retries=0,
        timeout=60,
    )
    all_results = {model: test_model(client, model) for model in MODELS}
    print_summary(all_results)


if __name__ == "__main__":
    main()
