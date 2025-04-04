from ...circuits.general_circuit import *

def measure_to_bits_example():
    test_circuit = Circuit(mode="array", verbose=True)
    array=QubitArray(q=4)
    test_circuit.upload_qubit_array(array)
    test_circuit.apply_gate(Hadamard, index=0)
    test_circuit.apply_gate(Hadamard, index=2)
    test_circuit.measure_states(index=0, basis="Z")
    test_circuit.measure_states(index=1, basis="X")
    test_circuit.measure_states(index=2, basis="Z")
    test_circuit.measure_states(index=3, basis="X")
    print(test_circuit.download_bits())

if __name__ == "__main__":
    measure_to_bits_example()