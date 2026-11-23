"""Обработчики Telegram-бота для диалога с языковой моделью."""

import asyncio
import contextlib
import logging

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.config import Settings
from app.prompts import MODES
from app.services.context import (
    ChatMessage,
    split_markdown_with_fallback,
    split_telegram_text,
    trim_history,
)
from app.services.llm_client import LLMClient, LLMError
from app.storage.postgres import ALLOWED_TEMPERATURES, Storage

logger = logging.getLogger(__name__)


def _settings_keyboard(current_temp: float) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            text=f"✓ {val:.1f}" if abs(current_temp - val) < 1e-9 else f"{val:.1f}",
            callback_data=f"set_temp:{val:.1f}",
        )
        for val in ALLOWED_TEMPERATURES
    ]
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


def _format_settings_text(mode: str, model: str, temperature: float) -> str:
    mode_spec = MODES.get(mode, MODES["study"])
    return (
        f"Текущий режим: {mode_spec.title} ({mode_spec.command})\n"
        f"Модель: {model}\n"
        f"Температура: {temperature:.1f}\n\n"
        "Выберите temperature кнопкой ниже или командой, например: /settings 0.3"
    )


def _parse_temperature(raw: str) -> float | None:
    try:
        parsed = float(raw.strip().replace(",", "."))
    except ValueError:
        return None
    for allowed in ALLOWED_TEMPERATURES:
        if abs(parsed - allowed) < 1e-9:
            return allowed
    return None


async def _keep_typing(bot: Bot, chat_id: int) -> None:
    while True:
        with contextlib.suppress(Exception):
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        await asyncio.sleep(4)


async def handle_start(message: Message) -> None:
    modes_lines = "\n".join(f"• {spec.command} — {spec.description}" for spec in MODES.values())
    text = (
        "Привет! Я AI-ассистент студента.\n\n"
        "Доступные режимы:\n"
        f"{modes_lines}\n\n"
        "Управление:\n"
        "• /settings — просмотр настроек и выбор temperature (0.0, 0.3, 0.7, 1.0)\n"
        "• /reset — очистка истории текущего диалога\n\n"
        "По умолчанию включён режим /study. Просто отправьте текстовый вопрос."
    )
    await message.answer(text, parse_mode=None)


async def handle_mode_switch(message: Message, storage: Storage) -> None:
    if message.from_user is None or message.text is None:
        return
    command_token = message.text.split()[0].lstrip("/").split("@")[0].lower()
    spec = MODES[command_token]
    await storage.set_mode_and_clear_history(message.from_user.id, spec.code)
    await message.answer(
        f"Режим переключён на {spec.command} ({spec.title}). История предыдущего диалога очищена.",
        parse_mode=None,
    )


async def handle_reset(message: Message, storage: Storage) -> None:
    if message.from_user is None:
        return
    await storage.clear_history(message.from_user.id)
    await message.answer(
        "История диалога очищена. Активный режим и temperature сохранены.",
        parse_mode=None,
    )


async def handle_settings(
    message: Message,
    command: CommandObject,
    storage: Storage,
    llm: LLMClient,
) -> None:
    if message.from_user is None:
        return
    user_id = message.from_user.id
    if command.args:
        parsed = _parse_temperature(command.args)
        if parsed is None:
            await message.answer(
                "Некорректное значение temperature. Допустимые значения: 0.0, 0.3, 0.7, 1.0.",
                parse_mode=None,
            )
            return
        await storage.set_temperature(user_id, parsed)
        user_cfg = await storage.get_settings(user_id)
        await message.answer(
            f"Температура обновлена: {parsed:.1f}\n\n"
            + _format_settings_text(user_cfg.mode, llm.model, user_cfg.temperature),
            reply_markup=_settings_keyboard(user_cfg.temperature),
            parse_mode=None,
        )
        return

    user_cfg = await storage.get_settings(user_id)
    await message.answer(
        _format_settings_text(user_cfg.mode, llm.model, user_cfg.temperature),
        reply_markup=_settings_keyboard(user_cfg.temperature),
        parse_mode=None,
    )


async def handle_settings_callback(
    callback: CallbackQuery,
    storage: Storage,
    llm: LLMClient,
) -> None:
    if callback.data is None or callback.message is None:
        return
    raw_value = callback.data.split(":", 1)[1]
    parsed = _parse_temperature(raw_value)
    if parsed is None:
        await callback.answer("Некорректное значение temperature.", show_alert=True)
        return

    user_id = callback.from_user.id
    await storage.set_temperature(user_id, parsed)
    user_cfg = await storage.get_settings(user_id)
    await callback.answer(f"Установлено: {parsed:.1f}")
    await callback.message.edit_text(
        _format_settings_text(user_cfg.mode, llm.model, user_cfg.temperature),
        reply_markup=_settings_keyboard(user_cfg.temperature),
        parse_mode=None,
    )


async def handle_unknown_command(message: Message) -> None:
    await message.answer(
        "Неизвестная команда. Отправьте /start, чтобы посмотреть список доступных команд.",
        parse_mode=None,
    )


async def handle_text_message(
    message: Message,
    storage: Storage,
    llm: LLMClient,
    settings: Settings,
) -> None:
    if message.from_user is None or message.text is None:
        return

    user_text = message.text.strip()
    if not user_text:
        return

    user_id = message.from_user.id
    user_cfg = await storage.get_settings(user_id)
    mode_spec = MODES.get(user_cfg.mode, MODES["study"])
    raw_history = await storage.get_history(user_id, limit=settings.history_max_messages)
    trimmed_history = trim_history(
        raw_history,
        current_user_text=user_text,
        max_messages=settings.history_max_messages,
        max_chars=settings.history_max_chars,
    )

    llm_messages = [
        ChatMessage(role="system", content=mode_spec.system_prompt),
        *trimmed_history,
        ChatMessage(role="user", content=user_text),
    ]

    typing_task: asyncio.Task[None] | None = None
    if message.bot is not None:
        typing_task = asyncio.create_task(_keep_typing(message.bot, message.chat.id))

    try:
        result = await llm.generate(llm_messages, temperature=user_cfg.temperature)
    except LLMError as exc:
        await message.answer(exc.user_message, parse_mode=None)
        return
    finally:
        if typing_task is not None:
            typing_task.cancel()
            await asyncio.gather(typing_task, return_exceptions=True)

    final_text = result.text
    if result.finish_reason == "length":
        final_text += "\n\n[Ответ был сокращён из-за ограничения максимальной длины генерации.]"

    await storage.save_turn(user_id, user_text, final_text)
    for raw_part, html_part in split_markdown_with_fallback(final_text):
        try:
            await message.answer(html_part, parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            logger.warning("Telegram HTML parse failed for chunk; falling back to plain text.")
            for plain_part in split_telegram_text(raw_part):
                await message.answer(plain_part, parse_mode=None)


def create_assistant_router() -> Router:
    r = Router(name="assistant")
    r.message.filter(F.chat.type == "private", F.text)
    r.callback_query.filter(F.message.chat.type == "private")
    r.message.register(handle_start, CommandStart())
    r.message.register(handle_mode_switch, Command("study", "translate", "review"))
    r.message.register(handle_reset, Command("reset"))
    r.message.register(handle_settings, Command("settings"))
    r.callback_query.register(handle_settings_callback, F.data.startswith("set_temp:"))
    r.message.register(handle_unknown_command, F.text.startswith("/"))
    r.message.register(handle_text_message, F.text)
    return r
