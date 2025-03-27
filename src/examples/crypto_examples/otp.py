from ...crypto_protocols import otp

message = "00110011001101010101"

key = otp.gen(message)
print(f"The randomly generated key: {key}")
cipher = otp.enc(key, message)
print(f"The encrypted message: {cipher}")
decrypted_message = otp.dec(key, cipher)
print(f"The decrypted message: {decrypted_message}")