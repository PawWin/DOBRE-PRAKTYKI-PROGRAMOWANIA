import pytest

from testy_jednostkowe_functions import (
    is_palindrome,
    fibonacci,
    count_vowels,
    calculate_discount,
    flatten_list,
)


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
        (-1, ValueError)
    ],
)
def test_fibonacci(n, expected):
    if expected == ValueError:
        with pytest.raises(ValueError):
            fibonacci(n)
    else:
        assert fibonacci(n) == expected

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Python", 2),
        ("AEIOUY", 6),
        ("bcd", 0),
        ("", 0),
        ("Próba żółwia", 5),
    ],
)
def test_count_vowels(text, expected):
    assert count_vowels(text) == expected

@pytest.mark.parametrize(
    "price, discount, expected",
    [
        (100.0, 0.2, 80.0),
        (50.0, 0, 50.0),
        (200.0, 1, 0.0),
        (100.0, -0.1, ValueError),
        (100.0, 1.5, ValueError),
    ],
)
def test_calculate_discount(price, discount, expected):
    if expected == ValueError:
        with pytest.raises(ValueError):
            calculate_discount(price, discount)
    else:
        assert calculate_discount(price, discount) == expected

@pytest.mark.parametrize(
    "nested_list, expected",
    [
        ([1, 2, 3], [1, 2, 3]),
        ([1, [2, 3], [4, [5]]], [1, 2, 3, 4, 5]),
        ([], []),
        ([[[1]]], [1]),
        ([1, [2, [3, [4]]]], [1, 2, 3, 4]),
    ],
)
def test_flatten_list(nested_list, expected):
    assert flatten_list(nested_list) == expected

