

def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome."""
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]

def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

