from ..circuit.classes import *
from ..circuit.classes.lightweight_circuit import *
from ..circuit_algorithms.grover_search import *
from ..crypto_protocols import *
from ..crypto_protocols import bb84
from ..crypto_protocols import otp
import numpy as np




message = "00110011001101010101"

key = otp.gen(message)
print(f"The randomly generated key: {key}")
cipher = otp.enc(key, message)
print(f"The encrypted message: {cipher}")
decrypted_message = otp.dec(key, cipher)
print(f"The decrypted message: {decrypted_message}")