from ...circuit_utilities.circuit_errors import QC_error

class QutritStateError(QC_error):
    """Error related to qutrits and their operations"""
    def __init__(self, message="Invalid qutrit operation or state"):
        self.message = message
        super().__init__(self.message)

class QutritStatePreparationError(QC_error):
    """Error related to quantum qutrit state preparation."""
    def __init__(self, message="Invalid State Preparation Operation"):
        self.message = message
        super().__init__(self.message)