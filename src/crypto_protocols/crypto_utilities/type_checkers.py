from .crypto_errors import PrimitiveError

def is_bit_string(string):
    return set(string).issubset({'0', '1'}) and len(string) > 0

def check_bit_string(string):
    if not isinstance(string, str):
        raise PrimitiveError(f"Inputted string cannot be of type {type(string)}, expected type str")
    if not is_bit_string(string):
        raise PrimitiveError(f"The inputted string must be composed of only binary '0' or '1', not of set {set(string)} of type {type(string)}" )