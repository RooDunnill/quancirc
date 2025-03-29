class QC_error(Exception):                 #a custom error class to raise custom errors
    """Creates my own custom errors defined in qc_dat."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"

class PrintError(QC_error):
    def __init__(self, message="Invalid Print Operation"):
        self.message = message
        super().__init__(self.message)

class SparseMatrixError(QC_error):
    def __init__(self, message="Invalid Sparse Matrix Operation"):
        self.message = message
        super().__init__(self.message)


