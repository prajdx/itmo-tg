"""Изолированный клиент OpenAI-совместимого LLM API."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

import aiohttp

from app.services.context import ChatMessage

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Безопасная ошибка обращения к LLM для ответа пользователю."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


class LLMClient(Protocol):
    model: str

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float,
    ) -> LLMResult: ...

    async def close(self) -> None: ...


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        max_tokens: int = 3500,
    ) -> None:
        self._api_key = api_key
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._max_tokens = max_tokens
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float,
    ) -> LLMResult:
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        logger.info("LLM request started request_id=%s model=%s", request_id, self.model)

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            session = self._get_session()
            async with session.post(self._endpoint, json=payload, headers=headers) as resp:
                if resp.status >= 400:
                    elapsed = int((time.perf_counter() - started) * 1000)
                    logger.warning(
                        "LLM API error request_id=%s model=%s status=%d duration_ms=%d",
                        request_id,
                        self.model,
                        resp.status,
                        elapsed,
                    )
                    raise LLMError(
                        "Сервис модели вернул ошибку и временно недоступен. Попробуйте позже."
                    )
                data = await resp.json()
        except LLMError:
            raise
        except TimeoutError as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            logger.warning(
                "LLM timeout request_id=%s model=%s duration_ms=%d",
                request_id,
                self.model,
                elapsed,
            )
            raise LLMError("Сервис модели не успел ответить вовремя. Попробуйте ещё раз.") from exc
        except (aiohttp.ClientError, ValueError) as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            logger.warning(
                "LLM network/parse error request_id=%s model=%s duration_ms=%d err=%s",
                request_id,
                self.model,
                elapsed,
                type(exc).__name__,
            )
            raise LLMError(
                "Не удалось связаться с сервисом модели. Попробуйте повторить запрос позже."
            ) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            logger.warning(
                "LLM unexpected error request_id=%s model=%s duration_ms=%d err=%s",
                request_id,
                self.model,
                elapsed,
                type(exc).__name__,
            )
            raise LLMError("Произошла ошибка при обращении к модели. Попробуйте позже.") from exc

        elapsed = int((time.perf_counter() - started) * 1000)
        choices = data.get("choices") if isinstance(data, dict) else None
        first = choices[0] if isinstance(choices, list) and choices else None
        message_obj = first.get("message") if isinstance(first, dict) else None
        raw_content = message_obj.get("content") if isinstance(message_obj, dict) else None
        text = raw_content.strip() if isinstance(raw_content, str) else ""
        finish_reason = first.get("finish_reason") if isinstance(first, dict) else None

        if not text:
            logger.warning(
                "LLM empty response request_id=%s model=%s finish_reason=%s duration_ms=%d",
                request_id,
                self.model,
                finish_reason,
                elapsed,
            )
            raise LLMError("Модель вернула пустой ответ. Попробуйте переформулировать вопрос.")

        usage = data.get("usage") if isinstance(data, dict) else None
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None

        logger.info(
            "LLM request finished request_id=%s model=%s finish_reason=%s "
            "prompt_tokens=%s completion_tokens=%s duration_ms=%d",
            request_id,
            self.model,
            finish_reason,
            prompt_tokens,
            completion_tokens,
            elapsed,
        )
        return LLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
