from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import os

def encrypt_string(key, plaintext):
    backend = default_backend()
    iv = os.urandom(16)  # Genera un vettore di inizializzazione casuale
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(128).padder()
    padded_plaintext = padder.update(plaintext.encode('utf-8')) + padder.finalize()
    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
    return iv + ciphertext

def decrypt_string(key, ciphertext):
    backend = default_backend()
    iv = ciphertext[:16]  # Estrae il vettore di inizializzazione dal ciphertext
    ciphertext = ciphertext[16:]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
    decryptor = cipher.decryptor()
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
    return plaintext.decode('utf-8')



# Esempio di utilizzo

key = b'supersegreto1234'  # Chiave di cifratura
#key = os.urandom(16)  # Genera una chiave casuale di 16 byte per AES

plaintext = "43 65 34 62"
ciphertext = encrypt_string(key, plaintext)
print("Ciphertext:", ciphertext)
print("Ciphertext length in byte:", len(ciphertext))
print("Ciphertext length in bit:", len(ciphertext) * 8)
print("Lunghezza plain text in byte:", len(plaintext))
decrypted_text = decrypt_string(key, ciphertext)
print("\nDecrypted text:", decrypted_text)
