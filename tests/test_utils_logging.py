import logging

from scanner.utils.logging import setup_logging


def test_setup_logging_info_level(caplog):
    setup_logging(verbose=False)
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_debug_level(caplog):
    setup_logging(verbose=True)
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_default_is_info():
    setup_logging()
    assert logging.getLogger().level == logging.INFO
