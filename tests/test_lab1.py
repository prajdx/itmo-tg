import logging
import sys
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

from app.config import ConfigError, Settings
from app.handlers.assistant import create_assistant_router
from app.logging_setup import SecretFilter
from app.prompts import MODES
from app.services.context import (
    ChatMessage,
    markdown_to_telegram_html,
    split_markdown_with_fallback,
    trim_history,
)
from app.services.llm_client import LLMError, LLMResult
from app.storage.postgres import DEFAULT_MODE, DEFAULT_TEMPERATURE, UserSettings

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"


class FakeStorage:
    def __init__(self) -> None:
        self.settings: dict[int, UserSettings] = {}
        self.history: dict[int, list[ChatMessage]] = {}

    async def ensure_schema(self) -> None:
        return None

    async def get_settings(self, user_id: int) -> UserSettings:
        if user_id not in self.settings:
            self.settings[user_id] = UserSettings(
                user_id=user_id,
                mode=DEFAULT_MODE,
                temperature=DEFAULT_TEMPERATURE,
            )
        return self.settings[user_id]

    async def set_mode_and_clear_history(self, user_id: int, mode: str) -> None:
        current = await self.get_settings(user_id)
        self.settings[user_id] = UserSettings(
            user_id=user_id,
            mode=mode,
            temperature=current.temperature,
        )
        self.history[user_id] = []

    async def set_temperature(self, user_id: int, temperature: float) -> None:
        current = await self.get_settings(user_id)
        self.settings[user_id] = UserSettings(
            user_id=user_id,
            mode=current.mode,
            temperature=temperature,
        )

    async def get_history(self, user_id: int, limit: int) -> list[ChatMessage]:
        return list(self.history.get(user_id, [])[-limit:])

    async def save_turn(self, user_id: int, user_text: str, assistant_text: str) -> None:
        items = self.history.setdefault(user_id, [])
        items.append(ChatMessage(role="user", content=user_text))
        items.append(ChatMessage(role="assistant", content=assistant_text))

    async def clear_history(self, user_id: int) -> None:
        self.history[user_id] = []


class FakeLLMClient:
    def __init__(
        self,
        responses: list[str] | None = None,
        error: Exception | None = None,
        finish_reason: str | None = "stop",
    ) -> None:
        self.model = "fake-model"
        self.calls: list[tuple[list[ChatMessage], float]] = []
        self._responses = list(responses or ["Готовый ответ"])
        self._error = error
        self._finish_reason = finish_reason

    async def generate(self, messages: list[ChatMessage], temperature: float) -> LLMResult:
        self.calls.append((list(messages), temperature))
        if self._error is not None:
            raise self._error
        text = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return LLMResult(text=text, finish_reason=self._finish_reason)

    async def close(self) -> None:
        return None


def make_settings(max_messages: int = 12, max_chars: int = 8000) -> Settings:
    return Settings(
        bot_token=TOKEN,
        postgres_password="secret",
        llm_api_key="llm-secret-key",
        llm_base_url="https://api.example.com/v1",
        llm_model="fake-model",
        history_max_messages=max_messages,
        history_max_chars=max_chars,
    )


def build_dispatcher() -> Dispatcher:
    dispatcher = Dispatcher()
    dispatcher.include_router(create_assistant_router())
    return dispatcher


def make_update(
    update_id: int,
    user_id: int,
    text: str,
    chat_type: str = "private",
) -> Update:
    message = Message(
        message_id=update_id,
        date=datetime.now(UTC),
        chat=Chat(id=user_id, type=chat_type),
        from_user=User(id=user_id, is_bot=False, first_name=f"User{user_id}"),
        text=text,
    )
    return Update(update_id=update_id, message=message)


def sent_messages(bot: Bot) -> list[SendMessage]:
    return [
        call.args[1]
        for call in bot.session.call_args_list
        if len(call.args) > 1 and isinstance(call.args[1], SendMessage)
    ]


async def test_1_message_passes_active_mode_prompt():
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dp = build_dispatcher()
    storage = FakeStorage()
    llm = FakeLLMClient(["Ответ про цикл событий"])
    cfg = make_settings()

    # Act
    await dp.feed_update(
        bot,
        make_update(1, 100, "Что такое event loop?"),
        storage=storage,
        llm=llm,
        settings=cfg,
    )

    # Assert
    assert len(llm.calls) == 1
    sent_to_llm, temp = llm.calls[0]
    assert temp == 0.3
    assert sent_to_llm[0] == ChatMessage(role="system", content=MODES["study"].system_prompt)
    assert sent_to_llm[-1] == ChatMessage(role="user", content="Что такое event loop?")
    assert [m.text for m in sent_messages(bot)] == ["Ответ про цикл событий"]


async def test_2_two_users_histories_do_not_mix():
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dp = build_dispatcher()
    storage = FakeStorage()
    llm = FakeLLMClient(["Ответ 1", "Ответ 2", "Ответ 3"])
    cfg = make_settings()

    # Act
    await dp.feed_update(
        bot, make_update(1, 101, "Секрет первого"), storage=storage, llm=llm, settings=cfg
    )
    await dp.feed_update(
        bot, make_update(2, 202, "Вопрос второго"), storage=storage, llm=llm, settings=cfg
    )
    await dp.feed_update(
        bot, make_update(3, 101, "Продолжим"), storage=storage, llm=llm, settings=cfg
    )

    # Assert
    second_user_msgs, _ = llm.calls[1]
    assert all("Секрет первого" not in m.content for m in second_user_msgs)
    third_call_msgs, _ = llm.calls[2]
    assert [m.role for m in third_call_msgs] == ["system", "user", "assistant", "user"]
    assert third_call_msgs[1].content == "Секрет первого"
    assert third_call_msgs[2].content == "Ответ 1"
    assert third_call_msgs[3].content == "Продолжим"


async def test_3_history_trimming_preserves_order_and_current_query():
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dp = build_dispatcher()
    storage = FakeStorage()
    await storage.save_turn(100, "старый-1", "ответ-1")
    await storage.save_turn(100, "средний-2", "ответ-2")
    await storage.save_turn(100, "свежий-3", "ответ-3")
    llm = FakeLLMClient(["Итог"])
    cfg = make_settings(max_messages=4, max_chars=26)

    # Act
    await dp.feed_update(bot, make_update(1, 100, "Новый"), storage=storage, llm=llm, settings=cfg)

    # Assert: "Новый"(5) + "ответ-3"(7) + "свежий-3"(8) = 20 <= 26, "ответ-2" не лезет
    msgs, _ = llm.calls[0]
    assert [m.role for m in msgs] == ["system", "user", "assistant", "user"]
    assert [m.content for m in msgs[1:]] == ["свежий-3", "ответ-3", "Новый"]


def test_3b_trim_history_drops_orphan_assistant_at_start():
    # Arrange
    history = [
        ChatMessage(role="user", content="Очень длинный вопрос " * 10),
        ChatMessage(role="assistant", content="Ок"),
    ]

    # Act
    trimmed = trim_history(history, current_user_text="Новый", max_messages=10, max_chars=30)

    # Assert
    assert trimmed == []


async def test_4_mode_switch_clears_history_and_saves_mode():
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dp = build_dispatcher()
    storage = FakeStorage()
    await storage.save_turn(100, "Старый вопрос", "Старый ответ")
    llm = FakeLLMClient(["Translated"])
    cfg = make_settings()

    # Act
    await dp.feed_update(
        bot, make_update(1, 100, "/translate"), storage=storage, llm=llm, settings=cfg
    )
    await dp.feed_update(
        bot,
        make_update(2, 100, "Переведи на английский: Привет"),
        storage=storage,
        llm=llm,
        settings=cfg,
    )

    # Assert
    assert (await storage.get_settings(100)).mode == "translate"
    assert len(llm.calls) == 1
    msgs, _ = llm.calls[0]
    assert msgs[0].content == MODES["translate"].system_prompt
    assert [m.content for m in msgs[1:]] == ["Переведи на английский: Привет"]
    assert "переключён" in sent_messages(bot)[0].text.lower()


async def test_5_reset_clears_only_calling_user_history_preserving_settings():
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dp = build_dispatcher()
    storage = FakeStorage()
    await storage.set_mode_and_clear_history(100, "review")
    await storage.set_temperature(100, 0.7)
    await storage.save_turn(100, "Мой код", "Ревью")
    await storage.save_turn(200, "Чужой вопрос", "Чужой ответ")
    llm = FakeLLMClient()
    cfg = make_settings()

    # Act
    await dp.feed_update(bot, make_update(1, 100, "/reset"), storage=storage, llm=llm, settings=cfg)

    # Assert
    assert await storage.get_history(100, 10) == []
    assert len(await storage.get_history(200, 10)) == 2
    assert (await storage.get_settings(100)).mode == "review"
    assert (await storage.get_settings(100)).temperature == 0.7
    assert (
        sent_messages(bot)[0].text
        == "История диалога очищена. Активный режим и temperature сохранены."
    )


async def test_6_settings_accepts_valid_and_rejects_invalid_temperature():
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dp = build_dispatcher()
    storage = FakeStorage()
    llm = FakeLLMClient()
    cfg = make_settings()

    # Act
    await dp.feed_update(
        bot, make_update(1, 100, "/settings 1.0"), storage=storage, llm=llm, settings=cfg
    )
    await dp.feed_update(
        bot, make_update(2, 100, "/settings 0.5"), storage=storage, llm=llm, settings=cfg
    )

    # Assert
    assert (await storage.get_settings(100)).temperature == 1.0
    assert llm.calls == []
    replies = sent_messages(bot)
    assert "Температура обновлена: 1.0" in replies[0].text
    assert (
        replies[1].text
        == "Некорректное значение temperature. Допустимые значения: 0.0, 0.3, 0.7, 1.0."
    )


async def test_7_llm_error_returns_safe_message_and_does_not_save_turn():
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dp = build_dispatcher()
    storage = FakeStorage()
    err = LLMError("Сервис модели не успел ответить вовремя. Попробуйте ещё раз.")
    llm = FakeLLMClient(error=err)
    cfg = make_settings()

    # Act
    await dp.feed_update(
        bot, make_update(1, 100, "Объясни стек"), storage=storage, llm=llm, settings=cfg
    )

    # Assert
    assert await storage.get_history(100, 10) == []
    replies = sent_messages(bot)
    assert len(replies) == 1
    assert replies[0].text == "Сервис модели не успел ответить вовремя. Попробуйте ещё раз."
    assert "llm-secret-key" not in replies[0].text


async def test_8_long_response_is_split_under_4096_without_data_loss():
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dp = build_dispatcher()
    storage = FakeStorage()
    long_text = ("А" * 3000) + "\n\n" + ("Б" * 3000)
    llm = FakeLLMClient([long_text])
    cfg = make_settings()

    # Act
    await dp.feed_update(
        bot, make_update(1, 100, "Напиши много"), storage=storage, llm=llm, settings=cfg
    )

    # Assert
    replies = sent_messages(bot)
    assert len(replies) == 2
    assert all(1 <= len(r.text) <= 4096 for r in replies)
    assert replies[0].text == "А" * 3000
    assert replies[1].text == "Б" * 3000


def test_9_markdown_formatting_converted_to_safe_html():
    # Arrange
    raw = (
        "### Заголовок\n"
        "**Важно** про `user_id` и *курсив*.\n"
        "- пункт с <script>\n"
        "```python\nif a < b:\n    print('ok')\n```"
    )

    # Act
    rendered = markdown_to_telegram_html(raw)

    # Assert
    assert "<b>Заголовок</b>" in rendered
    assert "<b>Важно</b>" in rendered
    assert "<code>user_id</code>" in rendered
    assert "<i>курсив</i>" in rendered
    assert "• пункт с &lt;script&gt;" in rendered
    assert "<pre><code>if a &lt; b:\n    print('ok')</code></pre>" in rendered


def test_10_long_markdown_splits_under_4096_with_closed_code_blocks():
    # Arrange
    long_code = "```python\n" + ("x = 1 + 2\n" * 500) + "```"
    raw = "### Начало\n" + long_code

    # Act
    parts = split_markdown_with_fallback(raw)

    # Assert
    assert len(parts) >= 2
    for _, html_part in parts:
        assert 1 <= len(html_part) <= 4096
        assert html_part.count("<pre><code>") == html_part.count("</code></pre>")


async def test_11_group_messages_are_ignored():
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dp = build_dispatcher()
    storage = FakeStorage()
    llm = FakeLLMClient(["Не должно вызываться"])
    cfg = make_settings()

    # Act
    await dp.feed_update(
        bot,
        make_update(1, -1001, "Привет группе", chat_type="group"),
        storage=storage,
        llm=llm,
        settings=cfg,
    )

    # Assert
    assert llm.calls == []
    assert sent_messages(bot) == []


async def test_12_truncated_response_is_marked():
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dp = build_dispatcher()
    storage = FakeStorage()
    llm = FakeLLMClient(["Частичный текст"], finish_reason="length")
    cfg = make_settings()

    # Act
    await dp.feed_update(
        bot, make_update(1, 100, "Расскажи всё"), storage=storage, llm=llm, settings=cfg
    )

    # Assert
    replies = sent_messages(bot)
    assert len(replies) == 1
    assert "ограничения максимальной длины генерации" in replies[0].text


def test_require_llm_fails_safely_when_missing(tmp_path):
    # Arrange
    env = tmp_path / ".env"
    env.write_text(f"BOT_TOKEN={TOKEN}\nPOSTGRES_PASSWORD=db\n", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        Settings.load(env, environ={}, require_llm=True)


def test_secret_filter_redacts_llm_api_key():
    # Arrange
    secret = "sk-very-secret-llm-key"
    secret_filter = SecretFilter([secret])
    try:
        raise RuntimeError(f"Header Bearer {secret}")
    except RuntimeError:
        record = logging.LogRecord(
            "test", logging.ERROR, "", 1, "Fail %s", (secret,), sys.exc_info()
        )

    # Act
    secret_filter.filter(record)
    rendered = logging.Formatter().format(record)

    # Assert
    assert secret not in rendered
