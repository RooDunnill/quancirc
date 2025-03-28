from ...circuit_utilities.circuit_errors import QC_error

class SymbQuantumStateError(QC_error):
    """Error related to symbolic qubits and their operations"""
    def __init__(self, message="Invalid symbolicqubit operation or state"):
        self.message = message
        super().__init__(self.message)

class SymbGateError(QC_error):
    """Error related to symbolic quantum gates and their operations"""
    def __init__(self, message="Invalid symbolic gate operation"):
        self.message = message
        super().__init__(self.message)

class SymbStatePreparationError(QC_error):
    """Error related to symbolic quantum state preparation."""
    def __init__(self, message="Invalid Symbolic State Preparation Operation"):
        self.message = message
        super().__init__(self.message)

class SymbQuantInfoError(QC_error):
    def __init__(self, message="Invalid Symbolic Quantum Information Operation"):
        self.message = message
        super().__init__(self.message)