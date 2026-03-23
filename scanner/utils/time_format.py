from __future__ import annotations


def format_time(seconds: float) -> str:
    """Convert a float number of seconds to an HH:MM:SS string.

    Fractional seconds are truncated (not rounded).

    >>> format_time(65.3)
    '00:01:05'
    >>> format_time(3661.9)
    '01:01:01'
    """
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
