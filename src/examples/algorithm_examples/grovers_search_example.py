from ...circuit_algorithms.grover_search import grover_search


def main():
    print(f"An example of Grovers with 16 random values and 16 qubits:")
    grover_search(16, n=16)

    print(f"An example of Grovers with 4 set oracle values, 10 iterations and 10 qubits:")
    oracle_values = [1,3,6,8]
    grover_search(oracle_values, iterations=10, n=10)

    print(f"An example of Grovers with 8 random oracle values, no set qubit amount and no set iterations")
    grover_search(8)

if __name__ == "__main__":
    main()
