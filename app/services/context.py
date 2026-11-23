"""Управление контекстом, безопасное HTML-форматирование и разбиение сообщений."""

import html
import re
from dataclasses import dataclass

TELEGRAM_MAX_LENGTH = 4096


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


def trim_history(
    history: list[ChatMessage],
    current_user_text: str,
    max_messages: int,
    max_chars: int,
) -> list[ChatMessage]:
    """Сокращает историю до max_messages и max_chars, сохраняя связность и текущий запрос."""
    budget_chars = max(0, max_chars - len(current_user_text))
    selected: list[ChatMessage] = []
    used_chars = 0

    for msg in reversed(history):
        if len(selected) >= max_messages:
            break
        msg_len = len(msg.content)
        if used_chars + msg_len > budget_chars:
            break
        selected.append(msg)
        used_chars += msg_len

    selected.reverse()
    while selected and selected[0].role != "user":
        selected.pop(0)
    return selected


def split_telegram_text(text: str, limit: int = TELEGRAM_MAX_LENGTH) -> list[str]:
    """Разбивает длинный текст на непустые части длиной <= limit без потери содержимого."""
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            parts.append(remaining)
            break

        chunk = remaining[:limit]
        split_at = max(chunk.rfind("\n\n"), chunk.rfind("\n"), chunk.rfind(" "))
        if split_at <= limit // 3:
            split_at = limit

        part = remaining[:split_at].rstrip()
        if not part:
            part = remaining[:limit]
            split_at = limit

        parts.append(part)
        remaining = remaining[split_at:].lstrip()

    return parts


def markdown_to_telegram_html(text: str) -> str:
    """Преобразует базовый Markdown ответа модели в безопасный Telegram HTML."""
    cleaned = text.replace("\x00", "")
    placeholders: list[str] = []

    def store(rendered: str) -> str:
        idx = len(placeholders)
        placeholders.append(rendered)
        return f"\x00CODE{idx}\x00"

    def replace_fenced(match: re.Match[str]) -> str:
        code = match.group(1).strip("\r\n")
        return store(f"<pre><code>{html.escape(code, quote=False)}</code></pre>")

    def replace_inline(match: re.Match[str]) -> str:
        code = match.group(1)
        return store(f"<code>{html.escape(code, quote=False)}</code>")

    cleaned = re.sub(r"```[^\n]*\n([\s\S]*?)```", replace_fenced, cleaned)
    cleaned = re.sub(r"`([^`\n]+)`", replace_inline, cleaned)

    escaped = html.escape(cleaned, quote=False)
    escaped = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", escaped, flags=re.MULTILINE)
    escaped = re.sub(r"\*\*([^\n]+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"__([^\n]+?)__", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?m)^(\s*)[-*]\s+", r"\1• ", escaped)
    escaped = re.sub(
        r"(?<!\*)\*(?!\s)([^\*\n]+?)(?<!\s)\*(?!\*)",
        r"<i>\1</i>",
        escaped,
    )

    for idx, value in enumerate(placeholders):
        escaped = escaped.replace(f"\x00CODE{idx}\x00", value)
    return escaped


def _inside_fenced_code(text: str) -> bool:
    return text.count("```") % 2 == 1


def split_markdown_with_fallback(
    text: str,
    limit: int = TELEGRAM_MAX_LENGTH,
) -> list[tuple[str, str]]:
    """Разбивает текст на пары (plain_part, html_part) с длиной html_part <= limit."""
    if not text:
        return []

    full_html = markdown_to_telegram_html(text)
    if len(full_html) <= limit:
        return [(text, full_html)]

    result: list[tuple[str, str]] = []
    remaining = text
    fence_prefix = ""

    while remaining:
        candidate_full = fence_prefix + remaining
        candidate_full_html = markdown_to_telegram_html(candidate_full)
        if len(candidate_full_html) <= limit:
            result.append((candidate_full, candidate_full_html))
            break

        window = min(len(remaining), limit)
        chosen: tuple[str, str, str] | None = None

        while window >= 1:
            chunk = remaining[:window]
            if window == len(remaining):
                split_at = window
            else:
                split_at = max(chunk.rfind("\n\n"), chunk.rfind("\n"), chunk.rfind(" "))
                if split_at <= window // 3:
                    split_at = window

                # Если разрез внутри блока кода, переносим блок в следующую часть.
                if _inside_fenced_code(fence_prefix + remaining[:split_at]):
                    last_fence = remaining[:split_at].rfind("\n```")
                    if last_fence > window // 3:
                        split_at = last_fence

            raw_piece = remaining[:split_at].rstrip()
            if not raw_piece:
                raw_piece = remaining[:split_at]

            part_text = fence_prefix + raw_piece
            next_prefix = ""
            if _inside_fenced_code(part_text):
                part_text = part_text + "\n```"
                next_prefix = "```\n"

            html_piece = markdown_to_telegram_html(part_text)
            if len(html_piece) <= limit:
                chosen = (part_text, html_piece, next_prefix)
                break

            window = max(1, split_at - max(100, window // 5))
            if window == 1 and len(markdown_to_telegram_html(fence_prefix + remaining[:1])) > limit:
                break

        if chosen is None:
            raw_piece = remaining[:1]
            part_text = fence_prefix + raw_piece
            html_piece = markdown_to_telegram_html(part_text)
            result.append((part_text, html_piece))
            remaining = remaining[1:]
            fence_prefix = ""
            continue

        part_text, html_piece, next_prefix = chosen
        result.append((part_text, html_piece))
        remaining = remaining[split_at:].lstrip("\n")
        fence_prefix = next_prefix

    return result
