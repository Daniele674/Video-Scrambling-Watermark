import mediapipe as mp
import numpy as np
import jpeglib
from imwatermark import WatermarkEncoder, WatermarkDecoder
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from reedsolo import RSCodec
import os
import io
import cv2
import subprocess
import time
import hashlib


# Classe per incapsulare lo stato di scrambling/descrambling
class ScrambleState:
    def __init__(self, seed=None, scramble_type=None, num_to_flip=None, face_region=None, key=None):
        self.seed = seed
        self.scramble_type = scramble_type
        self.num_to_flip = num_to_flip
        self.face_region = face_region
        self.key = key


max_retries = 3  # Numero massimo di tentativi
retry_delay = 0.5  # Secondi di attesa tra un tentativo e l'altro

rs = RSCodec(15)  # 10 ecc symbols

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()


def get_facial_landmarks(frame):
    """Funzione per rilevare i landmark facciali."""
    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    facelandmarks = []
    if result.multi_face_landmarks:  # Aggiunto controllo
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                facelandmarks.append([x, y])
    return np.array(facelandmarks, np.int32) if facelandmarks else None  # Evita errori su array vuoto


# Permutazione anche del coefficiente DC
def scramble_dct_block(block, seed):
    """Scramble dei coefficienti DCT all'interno di un blocco usando un seme fisso."""
    rng = np.random.default_rng(seed)  # Create a new Generator instance with the given seed
    perm = rng.permutation(block.size)  # Generate the permutation
    return block.ravel()[perm].reshape(block.shape)  # Apply the permutation and restore the original shape


def descramble_dct_block(block, seed):
    """Riordina i coefficienti DCT scrambleati utilizzando lo stesso seme."""
    rng = np.random.default_rng(seed)  # Create a new Generator instance with the given seed
    perm = rng.permutation(block.size)  # Recreate the same permutation
    inv_perm = np.argsort(perm)  # Compute the inverse permutation
    return block.ravel()[inv_perm].reshape(block.shape)  # Reorder and restore the original shape


# Funzioni di Encryption e Decryption
def encrypt_string(key, plaintext):
    backend = default_backend()
    iv = os.urandom(16)  # Genera un vettore di inizializzazione casuale
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(128).padder()
    padded_plaintext = padder.update(plaintext) + padder.finalize()
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
    return plaintext


def scrambleface(img, first_frame, key, state):
    if state.scramble_type not in ["signFlip", "permutation"]:
        raise ValueError("Il tipo deve essere 'signFlip' o 'permutation'")

    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise ValueError("Impossibile leggere l'immagine")

    landmarks = get_facial_landmarks(img)
    min_x = np.min(landmarks[:, 0]) // 8
    max_x = np.max(landmarks[:, 0]) // 8
    min_y = np.min(landmarks[:, 1]) // 8
    max_y = np.max(landmarks[:, 1]) // 8

    if first_frame and state.scramble_type == "signFlip":
        data_str = f"{state.num_to_flip} {min_x} {max_x} {min_y} {max_y}"
    else:
        data_str = f"{min_x} {max_x} {min_y} {max_y}"

    data_bytes = data_str.encode('utf-8')
    ciphertext = encrypt_string(key, data_bytes)
    encoded_ciphertext = rs.encode(ciphertext)

    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', encoded_ciphertext)
    img_encoded = encoder.encode(img, 'dwtDctSvd')
    for attempt in range(max_retries):
        success = cv2.imwrite('frame.jpg', img_encoded, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
        if success:
            break
        time.sleep(retry_delay)

    # print('Lunghezza del watermark:', len(encoded_ciphertext) * 8)

    rng = np.random.default_rng(state.seed)

    if state.scramble_type == "signFlip":
        image = jpeglib.read_dct('frame.jpg')
        for i in range(min_y, max_y):
            for j in range(min_x, max_x):
                matrix = image.Y[i, j]
                shape = matrix.shape
                indices = rng.choice(np.prod(shape), state.num_to_flip)
                coords = np.unravel_index(indices, shape)
                matrix[coords] *= -1
                image.Y[i, j] = matrix
        image.write_dct('output_scrambled.jpg')
        scrambled_image = cv2.imread('output_scrambled.jpg')
        return scrambled_image

    elif state.scramble_type == "permutation":
        image = jpeglib.read_dct('frame.jpg')
        for i in range(min_y, max_y):
            for j in range(min_x, max_x):
                image.Y[i, j] = scramble_dct_block(image.Y[i, j], state.seed)
        image.write_dct('output_scrambled.jpg')
        scrambled_image = cv2.imread('output_scrambled.jpg')
        return scrambled_image


def descrambleface(img, first_frame, key, state):
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise ValueError("Impossibile leggere l'immagine")

    decoder = WatermarkDecoder('bytes', 376)  # 376 = chiave di 16 byte + 15 byte di ECC
    watermark = decoder.decode(img, 'dwtDctSvd')
    ciphertext = watermark
    for attempt in range(max_retries):
        success = cv2.imwrite('frame.jpg', img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
        if success:
            break
        print(f"Tentativo {attempt + 1} fallito. Riprovo...")
        time.sleep(retry_delay)

    # print('Testo cifrato estratto:', ciphertext)
    ciphertext, _, errors = rs.decode(ciphertext)
    # print(list(errors))
    # print('Testo cifrato corretto:', ciphertext)
    extracted_data = decrypt_string(key, ciphertext)
    extracted_data = extracted_data.decode('utf-8')
    # print('Dati estratti:', extracted_data)

    data_list = extracted_data.split()

    if first_frame:
        if len(data_list) == 5:
            state.num_to_flip = int(data_list[0])
            state.scramble_type = "signFlip"
            state.face_region = [int(i) for i in data_list[1:]]
        else:
            state.scramble_type = "permutation"
            state.face_region = [int(i) for i in data_list]
    else:
        state.face_region = [int(i) for i in data_list]

    min_x, max_x, min_y, max_y = state.face_region

    rng = np.random.default_rng(state.seed)

    if state.scramble_type == "signFlip":
        scrambled_image = jpeglib.read_dct('frame.jpg')
        for i in range(min_y, max_y):
            for j in range(min_x, max_x):
                matrix = scrambled_image.Y[i, j]
                shape = matrix.shape
                indices = rng.choice(np.prod(shape), state.num_to_flip)
                coords = np.unravel_index(indices, shape)
                matrix[coords] *= -1
                scrambled_image.Y[i, j] = matrix
        scrambled_image.write_dct('output_descrambled.jpg')
        descrambled_image = cv2.imread('output_descrambled.jpg')
        return descrambled_image

    elif state.scramble_type == "permutation":
        scrambled_image = jpeglib.read_dct('frame.jpg')
        for i in range(min_y, max_y):
            for j in range(min_x, max_x):
                scrambled_image.Y[i, j] = descramble_dct_block(scrambled_image.Y[i, j], state.seed)
        scrambled_image.write_dct('output_descrambled.jpg')
        descrambled_image = cv2.imread('output_descrambled.jpg')
        return descrambled_image


# Funzione helper per eliminare file in sicurezza
def safe_remove(file_path):
    """Elimina il file specificato se esiste."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Errore nella rimozione di {file_path}: {e}")


# Funzione helper per creare il video utilizzando ffmpeg con parametri extra opzionali
def create_video_with_ffmpeg(fps, output_video_path, output_data, extra_args=None):
    command = [
        'ffmpeg',
        '-y',
        '-r', str(fps),
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-i', '-',
        '-vcodec', 'libx264'  # Per MPEG 1 usare 'mpeg1video'
    ]
    if extra_args:
        command.extend(extra_args)
    command.append(output_video_path)

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(input=output_data)
        if process.returncode != 0:
            error_message = stderr.decode('utf-8')
            raise RuntimeError(f"Errore FFmpeg (code {process.returncode}): {error_message}")
        else:
            print("Video creato con successo.")
    except Exception as e:
        print(f"Si è verificato un errore durante la creazione del video: {e}")


def extract_audio(input_video_path, audio_output_path):
    command = [
        'ffmpeg',
        '-i', input_video_path,
        '-map', 'a',
        '-q:a', '0',
        '-y',
        audio_output_path
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    if b'Stream #0:1' not in result.stderr:
        print("No audio stream found.")
        return False
    return True


def combine_audio_video(video_path, audio_path, output_path):
    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        output_path
    ]
    subprocess.run(command, check=True)


from tqdm import tqdm

from tqdm import tqdm


def scramblevideo(input_video_path, output_video_path=None, scramble_settings=None, key=None, progress_callback=None):
    if scramble_settings is None:
        scramble_settings = {
            'scramble_type': 'permutation',
            'num_to_flip': 0,
        }

    if len(key) not in [16, 24, 32]:
        print("La chiave deve essere lunga 16, 24 o 32 byte")
        exit(1)

    key_hash = hashlib.sha256(key).hexdigest()
    seed = int(key_hash, 16)
    # print(f"Seed: {seed}")

    audio_output_path = 'extracted_audio.aac'
    has_audio = extract_audio(input_video_path, audio_output_path)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Errore: impossibile aprire il video {input_video_path}.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    frames = []

    state = ScrambleState(
        seed=seed,
        scramble_type=scramble_settings['scramble_type'],
        num_to_flip=scramble_settings.get('num_to_flip', 0),
        key=key
    )

    with tqdm(total=total_frames, desc="Elaborazione frame", leave=False) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fine del video.")
                break

            frame_number += 1
            if progress_callback:
                progress_callback(frame_number, total_frames)

            try:
                first_frame = (frame_number == 1)
                scrambled_frame = scrambleface(frame, first_frame, state.key, state)
            except Exception as e:
                print(f"Errore nell'elaborazione del frame {frame_number}: {str(e)}")
                scrambled_frame = frame

            success, buffer = cv2.imencode('.jpg', scrambled_frame)
            if success:
                frames.append(buffer)
            else:
                print(f"Errore nella codifica del frame {frame_number}")

            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()
    print("Elaborazione video completata.")

    with io.BytesIO() as output:
        for frame in frames:
            output.write(frame.tobytes())
        output_data = output.getvalue()

    extra_args = ['-preset', 'slow', '-q:v', '0', '-crf', '10']
    create_video_with_ffmpeg(fps, output_video_path, output_data, extra_args)

    if has_audio:
        final_output_path = 'temp_output_video.mp4'
        combine_audio_video(output_video_path, audio_output_path, final_output_path)
        safe_remove(output_video_path)
        os.rename(final_output_path, output_video_path)
        safe_remove(audio_output_path)

    safe_remove('frame.jpg')
    safe_remove('output_scrambled.jpg')


def descramblevideo(input_video_path, output_video_path=None, key=None, progress_callback=None):
    if len(key) not in [16, 24, 32]:
        print("Chiave errata!")
        exit(1)

    key_hash = hashlib.sha256(key).hexdigest()
    seed = int(key_hash, 16)
    # print(f"Seed: {seed}")

    audio_output_path = 'extracted_audio.aac'
    has_audio = extract_audio(input_video_path, audio_output_path)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Errore: impossibile aprire il video {input_video_path}.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    frames = []

    state = ScrambleState(seed=seed, key=key)

    with tqdm(total=total_frames, desc="Elaborazione frame", leave=False) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if progress_callback:
                progress_callback(frame_number, total_frames)

            try:
                first_frame = (frame_number == 1)
                descrambled_frame = descrambleface(frame, first_frame, state.key, state)
            except Exception as e:
                print(f"Errore nell'elaborazione del frame {frame_number}: {str(e)}")
                descrambled_frame = frame

            success, buffer = cv2.imencode('.jpg', descrambled_frame)
            if success:
                frames.append(buffer)
            else:
                print(f"Errore nella codifica del frame {frame_number}")

            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()

    with io.BytesIO() as output:
        for frame in frames:
            output.write(frame.tobytes())
        output_data = output.getvalue()

    extra_args = ['-crf', '23']
    create_video_with_ffmpeg(fps, output_video_path, output_data, extra_args)

    if has_audio:
        final_output_path = 'temp_output_video.mp4'
        combine_audio_video(output_video_path, audio_output_path, final_output_path)
        safe_remove(output_video_path)
        os.rename(final_output_path, output_video_path)
        safe_remove(audio_output_path)

    safe_remove('frame.jpg')
    safe_remove('output_descrambled.jpg')
