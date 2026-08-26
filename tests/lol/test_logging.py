import logging
from logging.handlers import RotatingFileHandler

from oracle_bets_core.logger import (
    DEFAULT_LOG_BACKUPS,
    DEFAULT_LOG_MAX_BYTES,
    LOG_TOPIC,
    create_logger,
    instantiate_logger,
)


def test_create_logger_uses_single_rotating_file_handler(tmp_path):
    log_file = tmp_path / "data_pipeline.log"
    name = f"{LOG_TOPIC.DATA_PIPELINE.value}.test"

    logger = create_logger(name, log_file=log_file, max_bytes=1024, backup_count=2)
    same_logger = create_logger(name, log_file=log_file, max_bytes=1024, backup_count=2)

    try:
        handlers = logger.handlers
        assert logger is same_logger
        assert len(handlers) == 1
        assert isinstance(handlers[0], RotatingFileHandler)
        assert handlers[0].baseFilename.endswith("data_pipeline.log")
        assert "20" not in log_file.name[:4]
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        logging.Logger.manager.loggerDict.pop(name, None)


def test_application_topics_use_separate_bounded_handlers():
    data = instantiate_logger(LOG_TOPIC.DATA_PIPELINE)
    schedule = instantiate_logger(LOG_TOPIC.SCHEDULE_GENERATION)
    discord = instantiate_logger(LOG_TOPIC.DISCORD)

    assert data.handlers[0] is not schedule.handlers[0]
    assert schedule.handlers[0] is not discord.handlers[0]
    assert isinstance(data.handlers[0], RotatingFileHandler)
    assert isinstance(schedule.handlers[0], RotatingFileHandler)
    assert isinstance(discord.handlers[0], RotatingFileHandler)
    assert data.handlers[0].baseFilename.endswith("pipeline.log")
    assert schedule.handlers[0].baseFilename.endswith("schedule.log")
    assert discord.handlers[0].baseFilename.endswith("discord.log")
    assert data.handlers[0].maxBytes == DEFAULT_LOG_MAX_BYTES
    assert data.handlers[0].backupCount == DEFAULT_LOG_BACKUPS


def test_general_topic_remains_console_only():
    general = instantiate_logger(LOG_TOPIC.GENERAL)

    assert len(general.handlers) == 1
    assert not isinstance(general.handlers[0], RotatingFileHandler)
