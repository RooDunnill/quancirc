from ...circuit_utilities.circuit_errors import QC_error

class BaseQuantumStateError(QC_error):
    """Error related to qubits and their operations"""
    def __init__(self, message="Invalid qubit operation or state"):
        self.message = message
        super().__init__(self.message)

class BaseGateError(QC_error):
    """Error related to base quantum gates and their operations"""
    def __init__(self, message="Invalid base gate operation"):
        self.message = message
        super().__init__(self.message)

class BaseQuantumCircuitError(QC_error):
    """Error related to the base quantum circuit and their operations"""
    def __init__(self, message="Invalid base circuit operation"):
        self.message = message
        super().__init__(self.message)



class BaseStatePreparationError(QC_error):
    """Error related to base quantum state preparation."""
    def __init__(self, message="Invalid Base State Preparation Operation"):
        self.message = message
        super().__init__(self.message)


