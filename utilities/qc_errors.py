class QC_error(Exception):                 #a custom error class to raise custom errors from qc_dat
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

class GateError(QC_error):
    """Error related to quantum gates and their operations"""
    def __init__(self, message="Invalid gate operation"):
        self.message = message
        super().__init__(self.message)

class QuantumCircuitError(QC_error):
    """Error related to the quantum circuit and their operations"""
    def __init__(self, message="Invalid circuit operation"):
        self.message = message
        super().__init__(self.message)

class MeasurementError(QC_error):
    """Error related to quantum measurements."""
    def __init__(self, message="Measurement operation failed"):
        self.message = message
        super().__init__(self.message)

class StatePreparationError(QC_error):
    """Error related to quantum state preparation."""
    def __init__(self, message="Invalid State Preparation Operation"):
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

