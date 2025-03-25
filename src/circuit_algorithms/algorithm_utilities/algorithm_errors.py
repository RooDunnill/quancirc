class AlgorithmError(Exception):                 #a custom error class to raise custom errors from qc_dat
    """Creates my own custom errors defined in qc_dat."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"
    
class GroverSearchError(AlgorithmError):
    """Errors related to Grover Search Algorithm"""
    def __init__(self, message="Invalid Operation in Grovers Search Algorithm"):
        self.message = message
        super().__init__(self.message)