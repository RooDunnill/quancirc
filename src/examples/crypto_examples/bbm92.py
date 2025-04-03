from ...crypto_protocols import bbm92


bbm92_init_key = bbm92.gen_key(32)
print(f"The generated key is:\n{bbm92_init_key}")
bell_states = bbm92.gen_bell_states(bbm92_init_key)

qubits_a, key_a = bbm92.measure_a(bell_states)
print(f"Alice's initial key is:\n{key_a}")
qubits_b, key_b = bbm92.measure_b(bell_states)
print(f"Bob's initial key is:\n{key_b}")

reduced_key_a, reduced_key_b = bbm92.compare_basis(key_a, key_b)

print(f"Alice's reduced key:\n{reduced_key_a}")
print(f"Bob's reduced key:\n{reduced_key_b}")

