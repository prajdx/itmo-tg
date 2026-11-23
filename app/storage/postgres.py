"""Хранилище настроек и истории диалогов в PostgreSQL."""

from dataclasses import dataclass
from typing import Protocol

import asyncpg

from app.prompts import DEFAULT_MODE
from app.services.context import ChatMessage

DEFAULT_TEMPERATURE = 0.3
ALLOWED_TEMPERATURES = (0.0, 0.3, 0.7, 1.0)


@dataclass(frozen=True)
class UserSettings:
    user_id: int
    mode: str = DEFAULT_MODE
    temperature: float = DEFAULT_TEMPERATURE


class Storage(Protocol):
    async def ensure_schema(self) -> None: ...
    async def get_settings(self, user_id: int) -> UserSettings: ...
    async def set_mode_and_clear_history(self, user_id: int, mode: str) -> None: ...
    async def set_temperature(self, user_id: int, temperature: float) -> None: ...
    async def get_history(self, user_id: int, limit: int) -> list[ChatMessage]: ...
    async def save_turn(self, user_id: int, user_text: str, assistant_text: str) -> None: ...
    async def clear_history(self, user_id: int) -> None: ...


class PostgresStorage:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id BIGINT PRIMARY KEY,
                mode TEXT NOT NULL DEFAULT 'study',
                temperature DOUBLE PRECISION NOT NULL DEFAULT 0.3,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS dialog_messages (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_dialog_messages_user_id_id
                ON dialog_messages (user_id, id DESC);
            """
        )

    async def get_settings(self, user_id: int) -> UserSettings:
        row = await self._pool.fetchrow(
            """
            INSERT INTO user_settings (user_id, mode, temperature)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id) DO NOTHING
            """,
            user_id,
            DEFAULT_MODE,
            DEFAULT_TEMPERATURE,
        )
        _ = row
        record = await self._pool.fetchrow(
            "SELECT user_id, mode, temperature FROM user_settings WHERE user_id = $1",
            user_id,
        )
        assert record is not None
        return UserSettings(
            user_id=record["user_id"],
            mode=record["mode"],
            temperature=float(record["temperature"]),
        )

    async def set_mode_and_clear_history(self, user_id: int, mode: str) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                """
                INSERT INTO user_settings (user_id, mode, temperature)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id)
                DO UPDATE SET mode = EXCLUDED.mode, updated_at = NOW()
                """,
                user_id,
                mode,
                DEFAULT_TEMPERATURE,
            )
            await conn.execute("DELETE FROM dialog_messages WHERE user_id = $1", user_id)

    async def set_temperature(self, user_id: int, temperature: float) -> None:
        await self._pool.execute(
            """
            INSERT INTO user_settings (user_id, mode, temperature)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id)
            DO UPDATE SET temperature = EXCLUDED.temperature, updated_at = NOW()
            """,
            user_id,
            DEFAULT_MODE,
            temperature,
        )

    async def get_history(self, user_id: int, limit: int) -> list[ChatMessage]:
        rows = await self._pool.fetch(
            """
            SELECT role, content
            FROM (
                SELECT id, role, content
                FROM dialog_messages
                WHERE user_id = $1
                ORDER BY id DESC
                LIMIT $2
            ) recent
            ORDER BY id ASC
            """,
            user_id,
            limit,
        )
        return [ChatMessage(role=r["role"], content=r["content"]) for r in rows]

    async def save_turn(self, user_id: int, user_text: str, assistant_text: str) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO dialog_messages (user_id, role, content) VALUES ($1, 'user', $2)",
                user_id,
                user_text,
            )
            await conn.execute(
                "INSERT INTO dialog_messages (user_id, role, content) VALUES ($1, 'assistant', $2)",
                user_id,
                assistant_text,
            )

    async def clear_history(self, user_id: int) -> None:
        await self._pool.execute("DELETE FROM dialog_messages WHERE user_id = $1", user_id)
