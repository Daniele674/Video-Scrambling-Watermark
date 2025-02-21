from scramblery import scramblery as sc
import time

key = b'supersegreto1236'  # Chiave di cifratura (16, 24 o 32 byte)

# Configurazioni di scrambling (senza la chiave)
scramble_type = 'signFlip'  # 'signFlip' o 'permutation'

scramble_settings = {
    'scramble_type': scramble_type,
    'num_to_flip': 64,  # Numero di coefficienti da modificare (solo per 'signFlip')
}

video_input_path = 'testSet/volti.mp4'
video_scrambled_path = "output_video.mp4"
video_descrambled_path = "output_video_descrambled.mp4"

print("Inizio scrambling del video...")
# Misura il tempo di esecuzione
start_time = time.time()

sc.scramblevideo(video_input_path, video_scrambled_path, scramble_settings, key)

elapsed_time = time.time() - start_time
print(f"Tempo impiegato per lo scrambling: {elapsed_time:.2f} secondi")

start_time = time.time()

print("Inizio descrambling del video...")
# sc.descramblevideo(video_scrambled_path, video_descrambled_path, key)

elapsed_time = time.time() - start_time
print(f"Tempo impiegato per il descrambling: {elapsed_time:.2f} secondi")
print("Processo completato!")
