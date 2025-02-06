from scramblery import scramblery as sc
import time

key = b'supersegreto1234'   # Chiave di cifratura (16, 24 o 32 byte)

# Configurazioni di scrambling (senza la chiave)
scramble_type = 'permutation'  # 'signFlip' o 'permutation'
random_seed = 1234

scramble_settings = {
    'scramble_type': scramble_type,
    'num_to_flip': 64,  # Numero di coefficienti da modificare (solo per 'signFlip')
    'seed': random_seed
}

video_input_path = 'testSet/videoBreve.mp4'
video_scrambled_path = "output_video.mp4"
video_descrambled_path = "output_video_descrambled.mp4"

# Misura il tempo di esecuzione
start_time = time.time()

print("Inizio scrambling del video...")
# Passa la chiave come parametro separato
sc.scramblevideo(video_input_path, video_scrambled_path, scramble_settings, key)

elapsed_time = time.time() - start_time
print(f"Tempo impiegato per lo scrambling: {elapsed_time:.2f} secondi")

print("Inizio descrambling del video...")
sc.descramblevideo(video_scrambled_path, video_descrambled_path, key)
print("Processo completato!")
