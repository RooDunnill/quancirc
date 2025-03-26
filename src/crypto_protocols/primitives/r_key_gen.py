import random
from ..crypto_utilities.crypto_errors import PrimitiveError
from .k_hash_function import *

def n_length_bit_key(n):
    if isinstance(n, int):
        if n > 0:
            return "".join(random.choices("01", k=n))
        raise PrimitiveError(f"n cannot be {n}, must be positive")
    raise PrimitiveError(f"n cannot be type {type(n)}, expected type int")

def n_length_int_key(n):
    if isinstance(n, int):
        if n > 0:
            return "".join(random.choices("0123456789", k=n))
        raise PrimitiveError(f"n cannot be {n}, must be positive")
    raise PrimitiveError(f"n cannot be type {type(n)}, expected type int")

def n_length_hex_key(n):
    if isinstance(n, int):
        if n > 0:
            return "".join(random.choices("0123456789abcdef", k=n))
        raise PrimitiveError(f"n cannot be {n}, must be positive")
    raise PrimitiveError(f"n cannot be type {type(n)}, expected type int")

def n_length_letter_key(n):
    if isinstance(n, int):
        if n > 0:
            return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=n))
        raise PrimitiveError(f"n cannot be {n}, must be positive")
    raise PrimitiveError(f"n cannot be type {type(n)}, expected type int")