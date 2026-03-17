from __future__ import annotations

from openai import OpenAI

from thesis_bot.config import Settings


def create_openai_client(settings: Settings) -> OpenAI | None:
    if not settings.openai_api_key:
        return None
    return OpenAI(api_key=settings.openai_api_key)

