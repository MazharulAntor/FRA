"""Ask the configured LLM one question. Run from backend/:  python scripts/llm_smoke.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.llm.provider import get_llm

if __name__ == "__main__":
    settings = get_settings()
    print(f"provider={settings.llm_provider} model={settings.llm_model}")
    print(
        get_llm().complete(
            "You are a helpful assistant.", "Reply with the single word: ready"
        )
    )
