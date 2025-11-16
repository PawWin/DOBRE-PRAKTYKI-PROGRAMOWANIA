

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
