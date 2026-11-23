import argparse
import asyncio
import logging

from aiogram import Dispatcher

from app.config import ConfigError, Settings
from app.db import create_pool
from app.handlers.assistant import create_assistant_router
from app.health import HealthState, start_health_server
from app.logging_setup import configure_logging
from app.services.llm_client import OpenAICompatibleLLMClient
from app.storage.postgres import PostgresStorage
from app.telegram import create_bot

logger = logging.getLogger("app")


async def run(settings: Settings) -> None:
    state = HealthState()
    bot = create_bot(settings)
    llm_client: OpenAICompatibleLLMClient | None = None
    runner = None
    try:
        state.pool = await create_pool(settings)
        # Начальная проверка токена и маршрута через прокси ограничена по времени.
        async with asyncio.timeout(30):
            me = await bot.get_me()
            webhook = await bot.get_webhook_info()
        if webhook.url:
            raise ConfigError(
                "У бота установлен webhook. Удалите его перед запуском polling "
                "или используйте отдельного учебного бота."
            )
        storage = PostgresStorage(state.pool)
        await storage.ensure_schema()
        logger.info("PostgreSQL подключён: схема готова.")

        llm_client = OpenAICompatibleLLMClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        logger.info("Telegram доступен. Бот @%s запускает polling.", me.username)
        dispatcher = Dispatcher()
        dispatcher.include_router(create_assistant_router())
        runner = await start_health_server(state, settings.health_port)
        state.polling_task = asyncio.create_task(
            dispatcher.start_polling(
                bot,
                db=state.pool,
                storage=storage,
                llm=llm_client,
                settings=settings,
                allowed_updates=dispatcher.resolve_used_update_types(),
                close_bot_session=False,
            )
        )
        state.initialized = True
        await state.polling_task
    finally:
        state.initialized = False
        if state.polling_task and not state.polling_task.done():
            state.polling_task.cancel()
            await asyncio.gather(state.polling_task, return_exceptions=True)
        if runner:
            await runner.cleanup()
        if llm_client is not None:
            await llm_client.close()
        await bot.session.close()
        if state.pool:
            try:
                async with asyncio.timeout(10):
                    await state.pool.close()
            except TimeoutError:
                state.pool.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description="AI-ассистент студента в Telegram")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env_file, require_llm=True)
    except ConfigError as exc:
        print(str(exc))
        return 1
    configure_logging(settings)
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        logger.info("Бот остановлен.")
    except Exception:
        logger.exception("Не удалось запустить бот. Проверьте БД, токен, прокси и LLM.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
