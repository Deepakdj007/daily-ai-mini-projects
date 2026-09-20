"""Reference implementations for the two check-digit families.

Inputs:  a GSTIN or a 12-digit Verhoeff-protected number
Outputs: validity, and the check character a prefix should carry

These are the grader. Nothing here may import from the agent side of the
project, and nothing here may be clever: a reference implementation that is
hard to read is a reference implementation nobody checks.

Verhoeff is in the corpus for one reason that Luhn could not provide. Luhn
misses a 09 <-> 90 transposition; Verhoeff catches every single-digit error and
every adjacent transposition. A model that half-remembers the tables produces
something that validates its own examples and fails on a transposition, which
is exactly the overfit the held-out inputs are there to find.
"""

from __future__ import annotations

B36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# The dihedral group D5 multiplication table, the permutation table, and the
# inverse. Copied from the Verhoeff scheme as published; the whole point of the
# family is that these cannot be derived, only looked up.
_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def gstin_check_char(first14: str) -> str:
    """The 15th character of a GSTIN, from its first fourteen.

    Each character is read as a base-36 digit, multiplied by an alternating
    factor of 1 and 2, and the product is folded by adding its quotient and
    remainder against 36 before it joins the sum.
    """
    value = first14.strip().upper()
    if len(value) != 14 or any(char not in B36 for char in value):
        raise ValueError(f"need 14 base-36 characters, got {first14!r}")
    total = 0
    for position, char in enumerate(value):
        product = B36.index(char) * (1 if position % 2 == 0 else 2)
        quotient, remainder = divmod(product, 36)
        total += quotient + remainder
    return B36[(36 - total % 36) % 36]


def gstin_is_valid(gstin: str) -> bool:
    """True when a 15-character GSTIN carries the right check character."""
    value = gstin.strip().upper()
    if len(value) != 15 or any(char not in B36 for char in value):
        return False
    return gstin_check_char(value[:14]) == value[14]


def verhoeff_is_valid(number: str) -> bool:
    """True when a digit string ends in the Verhoeff check digit it should."""
    digits = number.strip().replace(" ", "")
    if not digits.isdigit():
        return False
    check = 0
    for position, char in enumerate(reversed(digits)):
        check = _D[check][_P[position % 8][int(char)]]
    return check == 0


def verhoeff_check_digit(prefix: str) -> int:
    """The digit that makes `prefix` validate.

    The position index is shifted by one against the validator, because the
    digit being computed will itself occupy position zero once appended.
    """
    digits = prefix.strip().replace(" ", "")
    if not digits.isdigit():
        raise ValueError(f"need digits, got {prefix!r}")
    check = 0
    for position, char in enumerate(reversed(digits)):
        check = _D[check][_P[(position + 1) % 8][int(char)]]
    return _INV[check]


if __name__ == "__main__":
    assert gstin_check_char("27AAPFU0939F1Z") == "V"
    assert gstin_is_valid("27AAPFU0939F1ZV")
    assert not gstin_is_valid("27AAPFU0939F1ZX")
    assert not gstin_is_valid("27AAPFU0939F1Z")          # too short
    # A check character that is a digit, not a letter - the case an
    # implementation assuming "the last character is a letter" gets wrong.
    tail = gstin_check_char("29AAGCB7383J1Z")
    assert tail.isdigit(), tail
    print(f"29AAGCB7383J1Z -> {tail} (a digit, not a letter)")

    assert verhoeff_check_digit("23456789012") == verhoeff_check_digit("23456789012")
    built = "23456789012" + str(verhoeff_check_digit("23456789012"))
    assert verhoeff_is_valid(built), built
    # Every adjacent transposition must break it. This is the property Luhn
    # does not have, and the reason this family is in the corpus.
    swapped = list(built)
    for i in range(len(swapped) - 1):
        if swapped[i] == swapped[i + 1]:
            continue
        pair = swapped[:]
        pair[i], pair[i + 1] = pair[i + 1], pair[i]
        assert not verhoeff_is_valid("".join(pair)), f"transposition at {i} survived"
    print(f"{built} valid, and every adjacent transposition of it is not")
    print("OK - ref_ids")
