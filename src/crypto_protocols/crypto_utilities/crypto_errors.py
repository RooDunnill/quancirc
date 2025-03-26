class CryptoError(Exception):                 #a custom error class to raise custom errors
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"
    
class PrimitiveError(CryptoError):
    def __init__(self, message="Invalid Operation in a Cryptography Primitive"):
        self.message = message
        super().__init__(self.message)

class BB84Error(CryptoError):
    def __init__(self, message="Invalid Operation in BB84 protocol"):
        self.message = message
        super().__init__(self.message)