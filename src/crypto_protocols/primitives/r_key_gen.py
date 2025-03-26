import random
from .crypto_utilities.crypto_errors import PrimitiveError

def n_bit_length_key(n):
    if isinstance(n, int):
        if n > 0:
            return "".join(random.choices("01", k=n))
        raise PrimitiveError(f"n cannot be {n}, must be positive")
    raise PrimitiveError(f"n cannot be type {type(n)}, expected type int")

def n_int_length_key(n):
    if isinstance(n, int):
        if n > 0:
            return "".join(random.choices("0123456789", k=n))
        raise PrimitiveError(f"n cannot be {n}, must be positive")
    raise PrimitiveError(f"n cannot be type {type(n)}, expected type int")

def n_hex_length_key(n):
    if isinstance(n, int):
        if n > 0:
            return "".join(random.choices("0123456789abcdef", k=n))
        raise PrimitiveError(f"n cannot be {n}, must be positive")
    raise PrimitiveError(f"n cannot be type {type(n)}, expected type int")
