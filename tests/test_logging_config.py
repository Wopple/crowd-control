"""Tests for logging configuration."""

from __future__ import annotations

import logging

from crowd_control.config import CrowdControlConfig
from crowd_control.logging_config import configure_logging

_PACKAGE_LOGGER = "crowd_control"


def _make_config(tmp_path, *, log_level="off"):
    return CrowdControlConfig(storage_dir=str(tmp_path), log_level=log_level)


def _get_logger():
    return logging.getLogger(_PACKAGE_LOGGER)


class TestInteractiveMode:
    def test_interactive_attaches_stderr_handler_at_info(self, tmp_path):
        config = _make_config(tmp_path)
        configure_logging(config, interactive=True)

        logger = _get_logger()
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1
        assert stream_handlers[0].level == logging.INFO

    def test_non_interactive_does_not_attach_stderr_handler(self, tmp_path):
        config = _make_config(tmp_path)
        configure_logging(config, interactive=False)

        logger = _get_logger()
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 0


class TestVerbose:
    def test_verbose_lowers_stderr_to_debug(self, tmp_path):
        config = _make_config(tmp_path)
        configure_logging(config, interactive=True, verbose=True)

        logger = _get_logger()
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1
        assert stream_handlers[0].level == logging.DEBUG


class TestFileHandler:
    def test_log_level_debug_attaches_file_handler(self, tmp_path):
        config = _make_config(tmp_path, log_level="debug")
        configure_logging(config, interactive=False)

        logger = _get_logger()
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].level == logging.DEBUG

    def test_log_level_off_does_not_attach_file_handler(self, tmp_path):
        config = _make_config(tmp_path, log_level="off")
        configure_logging(config, interactive=False)

        logger = _get_logger()
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0

    def test_file_handler_creates_log_file(self, tmp_path):
        config = _make_config(tmp_path, log_level="info")
        configure_logging(config, interactive=False)

        log_path = tmp_path / "logs" / "crowd-control.log"
        assert log_path.exists()


class TestNullHandlerFallback:
    def test_no_handlers_adds_null_handler(self, tmp_path):
        config = _make_config(tmp_path, log_level="off")
        configure_logging(config, interactive=False)

        logger = _get_logger()
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.NullHandler)
