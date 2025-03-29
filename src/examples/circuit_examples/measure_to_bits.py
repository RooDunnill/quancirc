from ...circuits.base_circuit import *

def measure_to_bits_example():
    test_circuit = Circuit(mode="array", verbose=True)
    array=QubitArray(q=4)
    test_circuit.upload_qubit_array(array)
    test_circuit.apply_gate_on_array(Hadamard, 0)
    test_circuit.apply_gate_on_array(Hadamard, 2)
    test_circuit.measure_states_on_array(0, basis="Z")
    test_circuit.measure_states_on_array(1, basis="X")
    test_circuit.measure_states_on_array(2, basis="Z")
    test_circuit.measure_states_on_array(3, basis="X")
    print(test_circuit.download_bits())

if __name__ == "__main__":
    measure_to_bits_example()