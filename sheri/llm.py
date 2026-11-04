import logging

from openai import AsyncOpenAI

from .config import Config

log = logging.getLogger(__name__)


class RouterAI:
    def __init__(self, config: Config):
        self.config = config
        self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    async def complete(self, messages: list[dict], tools: list[dict]) -> tuple[dict, int]:
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            tools=tools,
            # Без обычного текста: модель каждый раз явно выбирает say или stay_silent.
            tool_choice="required",
            max_completion_tokens=self.config.max_completion_tokens,
            reasoning_effort=self.config.reasoning_effort,
        )
        reply = response.choices[0].message
        usage = response.usage
        details = usage.prompt_tokens_details
        log.info(
            "токены: вход %s (из кэша %s), выход %s",
            usage.prompt_tokens,
            details.cached_tokens if details else "?",
            usage.completion_tokens,
        )
        message: dict = {"role": "assistant", "content": reply.content or ""}
        if reply.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in reply.tool_calls
            ]
        return message, usage.prompt_tokens
