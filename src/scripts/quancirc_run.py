from ..circuit.classes import *
from ..circuit.classes.lightweight_circuit import *
from ..circuit_algorithms.grover_search import *


test_qubit_array = QubitArray(name="test")
print(len(test_qubit_array))
test_qubit_array.add_qubit(q0)
test_qubit_array.add_qubit(q1)
test_qubit_array.add_qubit(qm)
test_qubit_array.add_qubit(qp)
print(len(test_qubit_array))
test_qubit_array.qubit_info(1)
test_qubit_array.pop_first_qubit()
print(len(test_qubit_array))
test_qubit_array.qubit_info(0)
test_qubit_array.insert_qubit(qpi, 0)
test_qubit_array.qubit_info(1)
print(len(test_qubit_array))
test_qubit_array.validate_array()
qubit_array_circuit = Circuit()
qubit_array_circuit.upload_qubit_array(test_qubit_array)
qubit_array_circuit.apply_gate_on_array(Hadamard, 2)
qubit_array_circuit.get_array_info()
test_qubit_array = qubit_array_circuit.download_qubit_array()
