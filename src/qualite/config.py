"""Central settings: paths, model name, API key.

WHY one module: every other file imports its settings from here, so changing a model
or a folder means editing one place. WHY read the environment: the same code works with
a local .env file, in Docker, and on a cloud service.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


@dataclass(frozen=True)
class Settings:
    """Project settings.

    WHY frozen: no module can change a setting by accident while the program runs.
    WHY repr=False: printing the settings must never reveal the key.
    """

    groq_api_key: str = field(repr=False)
    groq_base_url: str
    model: str


def load_settings() -> Settings:
    """Build the settings, and stop early if the key is missing.

    WHY stop here: a clear error now is easier to fix than a confusing API error later.
    """
    load_dotenv(ROOT / ".env")

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing: add it to the .env file at the project root.")

    return Settings(
        groq_api_key=key,
        groq_base_url=os.environ.get("GROQ_BASE_URL", DEFAULT_BASE_URL),
        model=os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
    )


if __name__ == "__main__":
    settings = load_settings()
    print(settings)
    print("API key found: yes")
