"""Custom tools the code agent can call from its generated Python.

A tool is just a typed function wrapped in @tool. smolagents reads the type
hints and the Args: docstring section to build the schema it shows the model,
so both are required — a missing arg description raises at agent build time.
"""

from smolagents import tool


@tool
def prime_factors(n: int) -> list[int]:
    """Return the prime factors of a positive integer, smallest first.

    Use this for exact integer factorisation instead of writing your own loop;
    it is reliable for large numbers and avoids off-by-one mistakes.

    Args:
        n: The positive integer to factorise. Must be 2 or greater.
    """
    if n < 2:
        raise ValueError("n must be 2 or greater")

    factors: list[int] = []
    divisor = 2
    remaining = n
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 1
    if remaining > 1:
        factors.append(remaining)
    return factors
