from .base_class_errors import *

def base_quant_state_validation(quant_state):
    if not isinstance(quant_state.history, list):
        raise BaseStatePreparationError(f"self.history cannot be of type {type(quant_state.history)}, expected type list")
    if not isinstance(quant_state.display_mode, str):
        raise BaseStatePreparationError(f"The inputted self.display_mode cannot be of type {type(quant_state.display_mode)}, expected type str")
    if not isinstance(quant_state.skip_val, bool):
        raise BaseStatePreparationError(f"self.skip_val cannot be of type {type(quant_state.skip_val)}, expected type bool")
    if not isinstance(quant_state.id, int) and not isinstance(quant_state.id, str):
        raise BaseStatePreparationError(f"self.id cannot be of type {type(quant_state.id)}, expected type int or str")