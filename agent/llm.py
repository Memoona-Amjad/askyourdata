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


class LLMError(Exception):
    """A friendly, user-facing LLM problem (missing key, quota hit, outage...)."""


def _setting(name: str, default: str = "") -> str:
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
    return _setting("LLM_MODEL", "gemini-3.6-flash")


@lru_cache(maxsize=1)
def _client():
    from google import genai

    api_key = _setting("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("No Gemini API key found. Add GEMINI_API_KEY to your .env file or Streamlit secrets.")
    return genai.Client(api_key=api_key)


def generate(prompt: str, system: str | None = None, retries: int = 3) -> str:
    """Send one prompt to Gemini and return the text reply.

    Retries a few times on temporary problems (rate limit bursts, overloaded server).
    """
    from google.genai import errors, types

    config = types.GenerateContentConfig(system_instruction=system) if system else None
    delay = 4.0
    for attempt in range(1, retries + 1):
        try:
            response = _client().models.generate_content(model=model_name(), contents=prompt, config=config)
            return (response.text or "").strip()
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            temporary = code in (429, 500, 503, 504)
            if temporary and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            if code == 429:
                raise LLMError("The AI service is busy or the free quota is used up. Please try again in a minute.") from exc
            if code in (400, 401, 403):
                raise LLMError("The AI service rejected the request. Check that the Gemini API key and model name are valid.") from exc
            raise LLMError("The AI service had a temporary problem. Please try again.") from exc
        except LLMError:
            raise
        except Exception as exc:  # network errors etc.
            if attempt < retries:
                time.sleep(delay)
                continue
            raise LLMError("Could not reach the AI service. Please check the internet connection and try again.") from exc
    raise LLMError("The AI service did not respond.")
