import logging
from logging.handlers import RotatingFileHandler

from oracle_bets_core.logger import LOG_TOPIC, create_logger


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
