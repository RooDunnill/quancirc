class QC_error(Exception):                 #a custom error class to raise custom errors
    """Creates my own custom errors defined in qc_dat."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"
    
class QuantumStateError(QC_error):
    """Error related to qubits and their operations"""
    def __init__(self, message="Invalid qubit operation or state"):
        self.message = message
        super().__init__(self.message)

class SymbQuantumStateError(QC_error):
    """Error related to symbolic qubits and their operations"""
    def __init__(self, message="Invalid symbolicqubit operation or state"):
        self.message = message
        super().__init__(self.message)

class LWQuantumStateError(QC_error):
    """Error related to lightweight qubits and their operations"""
    def __init__(self, message="Invalid lightweight qubit operation or state"):
        self.message = message
        super().__init__(self.message)

class BitError(QC_error):
    """Error related to bits and their operations"""
    def __init__(self, message="Invalid bit operation or state"):
        self.message = message
        super().__init__(self.message)

class GateError(QC_error):
    """Error related to quantum gates and their operations"""
    def __init__(self, message="Invalid gate operation"):
        self.message = message
        super().__init__(self.message)

class SymbGateError(QC_error):
    """Error related to symbolic quantum gates and their operations"""
    def __init__(self, message="Invalid symbolic gate operation"):
        self.message = message
        super().__init__(self.message)

class QuantumCircuitError(QC_error):
    """Error related to the quantum circuit and their operations"""
    def __init__(self, message="Invalid circuit operation"):
        self.message = message
        super().__init__(self.message)

class LWQuantumCircuitError(QC_error):
    """Error related to the lightweight quantum circuit and their operations"""
    def __init__(self, message="Invalid lightweight circuit operation"):
        self.message = message
        super().__init__(self.message)

class MeasureError(QC_error):
    """Error related to quantum measurements."""
    def __init__(self, message="Measurement operation failed"):
        self.message = message
        super().__init__(self.message)

class StatePreparationError(QC_error):
    """Error related to quantum state preparation."""
    def __init__(self, message="Invalid State Preparation Operation"):
        self.message = message
        super().__init__(self.message)

class SymbStatePreparationError(QC_error):
    """Error related to symbolic quantum state preparation."""
    def __init__(self, message="Invalid Symbolic State Preparation Operation"):
        self.message = message
        super().__init__(self.message)

class LWStatePreparationError(QC_error):
    """Error related to lightweight quantum state preparation."""
    def __init__(self, message="Invalid Lightweight State Preparation Operation"):
        self.message = message
        super().__init__(self.message)

class PrintError(QC_error):
    def __init__(self, message="Invalid Print Operation"):
        self.message = message
        super().__init__(self.message)

class QuantInfoError(QC_error):
    def __init__(self, message="Invalid Quantum Information Operation"):
        self.message = message
        super().__init__(self.message)

class SymbQuantInfoError(QC_error):
    def __init__(self, message="Invalid Symbolic Quantum Information Operation"):
        self.message = message
        super().__init__(self.message)

class SparseMatrixError(QC_error):
    def __init__(self, message="Invalid Sparse Matrix Operation"):
        self.message = message
        super().__init__(self.message)

class QubitArrayError(QC_error):
    def __init__(self, message="Invalid Qubit Array Operation"):
        self.message = message
        super().__init__(self.message)
