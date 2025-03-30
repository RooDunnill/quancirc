from ...circuit_utilities.circuit_errors import QC_error

class LwStatePreparationError(QC_error):
    """Error related to lightweight quantum state preparation."""
    def __init__(self, message="Invalid Lightweight State Preparation Operation"):
        self.message = message
        super().__init__(self.message)

class LwQuantumCircuitError(QC_error):
    """Error related to the lightweight quantum circuit and their operations"""
    def __init__(self, message="Invalid lightweight circuit operation"):
        self.message = message
        super().__init__(self.message)

class LwQuantumStateError(QC_error):
    """Error related to lightweight qubits and their operations"""
    def __init__(self, message="Invalid lightweight qubit operation or state"):
        self.message = message
        super().__init__(self.message)

class LwMeasureError(QC_error):
    def __init__(self, message="Invalid lightweight mesaure operation"):
        self.message = message
        super().__init__(self.message)