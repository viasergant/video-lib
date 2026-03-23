from scanner.utils.time_format import format_time


def test_format_time_basic():
    assert format_time(65.3) == "00:01:05"


def test_format_time_zero():
    assert format_time(0.0) == "00:00:00"


def test_format_time_exact_hour():
    assert format_time(3600.0) == "01:00:00"


def test_format_time_over_one_hour():
    assert format_time(3661.9) == "01:01:01"


def test_format_time_truncates_fractional():
    # 59.999 should give 00:00:59, not 00:01:00
    assert format_time(59.999) == "00:00:59"


def test_format_time_large_value():
    # 2h 30m 45s = 9045s
    assert format_time(9045.0) == "02:30:45"
