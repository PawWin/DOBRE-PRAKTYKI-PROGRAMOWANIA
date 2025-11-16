import pytest

from testy_jednostkowe_functions import is_palindrome, fibonacci


@pytest.mark.parametrize(
    "text, expected",
    [
        ("kajak", True),
        ("Kobyła ma mały bok", True),
        ("python", False),
        ("", True),
        ("A", True),
    ],
)
def test_is_palindrome(text, expected):
    assert is_palindrome(text) is expected


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, 0),
        (1, 1),
        (5, 5),
        (10, 55),
        (20, 6765),
        (-1, ValueError)
    ],
)
def test_fibonacci(n, expected):
    if expected == ValueError:
        with pytest.raises(ValueError):
            fibonacci(n)
    else:
        assert fibonacci(n) == expected
