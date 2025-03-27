from .primitives import *
from .crypto_utilities.crypto_errors import OTPError
from .crypto_utilities.type_checkers import check_bit_string

def gen(message):
    check_bit_string(message)
    n = len(message)
    return n_length_bit_key(n)

def enc(key, message):
    check_bit_string(key)
    check_bit_string(message)
    if len(key) != len(message):
        raise OTPError(f"The inputted key and message must be the same length, not of length {len(key)} and {len(message)}")
    return ''.join(str(int(k) ^ int(m)) for k, m in zip(key, message))

def dec(key, cipher):
    check_bit_string(key)
    check_bit_string(cipher)
    if len(key) != len(cipher):
        raise OTPError(f"The inputted key and message must be the same length, not of length {len(key)} and {len(cipher)}")
    return ''.join(str(int(k) ^ int(m)) for k, m in zip(key, cipher))