
def is_bit_string(string):
    return all(c in '01' for c in string) and isinstance(string, str) and len(string) > 0