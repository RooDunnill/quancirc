from ...circuit.classes.circuit import *
from ...circuit.classes.qubit_array import *
from ...circuit.classes.gate import * 

test_circuit = Circuit(mode="array", verbose=True)
array=QubitArray(q=4)
test_circuit.upload_qubit_array(array)
test_circuit.apply_gate_on_array(Hadamard, 0)
test_circuit.apply_gate_on_array(Hadamard, 2)
test_circuit.measure_states_on_array(0, "Z")
test_circuit.measure_states_on_array(1, "X")
test_circuit.measure_states_on_array(2, "Z")
test_circuit.measure_states_on_array(3, "X")
print(test_circuit.return_bits())