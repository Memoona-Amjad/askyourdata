"""The ONE place that talks to the LLM provider (Google Gemini).

Everything else calls `generate(prompt, system=...)` and gets plain text back,
so swapping to another provider only means changing this file.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

REQUEST_TIMEOUT_MS = 60_000


class LLMError(Exception):
    """A friendly, user-facing LLM problem (missing key, quota hit, outage...)."""


def setting(name: str, default: str = "") -> str:
    """Read a setting from the environment / .env first, then Streamlit secrets."""
    value = os.getenv(name, "")
    if value:
        return value
    try:
        import streamlit as st

        return str(st.secrets.get(name, default))
    except Exception:
        return default


def model_name() -> str:
    return setting("LLM_MODEL", "gemini-3.5-flash-lite")


def fallback_model_name() -> str:
    """Used when the main model is overloaded (HTTP 503) or rate limited (429)."""
    return setting("LLM_FALLBACK_MODEL", "gemini-3.1-flash-lite")


@lru_cache(maxsize=1)
def _client():
    from google import genai
    from google.genai import types

    api_key = setting("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("No Gemini API key found. Add GEMINI_API_KEY to your .env file or Streamlit secrets.")
    # 60 s timeout so a stuck request fails (and falls back) instead of hanging forever
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))


TEMPORARY_CODES = (429, 500, 503, 504)


def generate(prompt: str, system: str | None = None, retries: int = 2) -> str:
    """Send one prompt to Gemini and return the text reply.

    On temporary problems (overloaded server, rate-limit burst) it retries once,
    then switches to the fallback model, so the demo keeps working.
    """
    from google.genai import errors, types

    config = types.GenerateContentConfig(system_instruction=system) if system else None
    models = list(dict.fromkeys([model_name(), fallback_model_name()]))  # unique, in order
    last_code: int | None = None

    for model in models:
        delay = 2.0
        for attempt in range(1, retries + 1):
            try:
                response = _client().models.generate_content(model=model, contents=prompt, config=config)
                return (response.text or "").strip()
            except errors.APIError as exc:
                last_code = getattr(exc, "code", None)
                if last_code not in TEMPORARY_CODES:
                    if last_code in (400, 401, 403, 404):
                        raise LLMError("The AI service rejected the request. Check that the Gemini API key "
                                       "and model name are valid.") from exc
                    raise LLMError("The AI service had a problem with this request. Please try again.") from exc
                if attempt < retries:
                    time.sleep(delay)
                    delay *= 2
            except LLMError:
                raise
            except Exception as exc:  # network errors etc.
                if attempt == retries and model == models[-1]:
                    raise LLMError("Could not reach the AI service. Please check the internet "
                                   "connection and try again.") from exc
                time.sleep(delay)
        # this model kept failing with a temporary error -> try the next one

    if last_code == 429:
        raise LLMError("The free AI quota is used up for now. Please try again in a minute.")
    raise LLMError("The AI service is very busy right now. Please try again in a minute.")
