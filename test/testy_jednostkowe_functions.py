
import re
from collections import Counter


def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome."""
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]

def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (0-indexed)."""
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def count_vowels(s: str) -> int:
    """Count the number of vowels in a string (case-insensitive, incl. Polish chars)."""
    vowels = 'aeiouyąęóAEIOUYĄĘÓ'
    return sum(1 for char in s if char in vowels)

def calculate_discount(price: float, discount: float) -> float:
    """Return the price after applying discount in [0, 1]; raise for invalid discount."""
    if discount < 0 or discount > 1:
        raise ValueError("discount must be between 0 and 1")
    return price * (1 - discount)

def flatten_list(nested_list):
    """Flatten a nested list into a single-level list."""
    flat = []
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat


def word_frequency(text: str) -> dict:
    """Return a dictionary with word frequencies, case-insensitive and ignoring punctuation."""
    words = re.findall(r"\w+", text.lower(), re.UNICODE)
    return dict(Counter(words))

def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
