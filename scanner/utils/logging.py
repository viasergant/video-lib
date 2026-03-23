from __future__ import annotations

import logging


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger for the scanner.

    Args:
        verbose: When True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        force=True,
    )
