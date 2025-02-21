import mediapipe as mp
import numpy as np
import jpeglib
from imwatermark import WatermarkEncoder, WatermarkDecoder
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from reedsolo import RSCodec, ReedSolomonError
import os
import io
import cv2
import subprocess
import time
import hashlib
from tqdm import tqdm


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
    """Function to detect facial landmarks."""
    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=5,
                               min_detection_confidence=0.3,
                               min_tracking_confidence=0.3) as face_mesh:
        result = face_mesh.process(frame_rgb)

    facelandmarks_list = []
    if result.multi_face_landmarks:
        for facial_landmarks in result.multi_face_landmarks:
            facelandmarks = []
            for i in range(min(468, len(facial_landmarks.landmark))):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                facelandmarks.append([x, y])
            facelandmarks_list.append(np.array(facelandmarks, np.int32))

    return facelandmarks_list


def scramble_sign_flip(image, min_x, max_x, min_y, max_y, num_to_flip, seed):
    rng = np.random.default_rng(seed)
    for i in range(min_y, max_y):
        for j in range(min_x, max_x):
            matrix = image.Y[i, j]
            shape = matrix.shape
            indices = rng.choice(np.prod(shape), num_to_flip)
            coords = np.unravel_index(indices, shape)
            matrix[coords] *= -1
            matrix = np.clip(matrix, -1023, 1023)  # Ensure coefficients are within valid range
            image.Y[i, j] = matrix
    return image


def descramble_sign_flip(image, min_x, max_x, min_y, max_y, num_to_flip, seed):
    rng = np.random.default_rng(seed)
    for i in range(min_y, max_y):
        for j in range(min_x, max_x):
            matrix = image.Y[i, j]
            shape = matrix.shape
            indices = rng.choice(np.prod(shape), num_to_flip)
            coords = np.unravel_index(indices, shape)
            matrix[coords] *= -1
            matrix = np.clip(matrix, -1023, 1023)  # Ensure coefficients are within valid range
            image.Y[i, j] = matrix
    return image


def scramble_permutation(image, min_x, max_x, min_y, max_y, seed):
    rng = np.random.default_rng(seed)  # Initialize the random number generator
    block_size = image.Y[min_y, min_x].size  # Determine the size of a block
    perm = rng.permutation(block_size)  # Generate a single permutation

    for i in range(min_y, max_y):
        for j in range(min_x, max_x):
            block = image.Y[i, j]
            block = block.ravel()[perm].reshape(block.shape)  # Apply the same permutation
            block = np.clip(block, -1023, 1023)  # Ensure coefficients are within valid range
            image.Y[i, j] = block

    return image


def descramble_permutation(image, min_x, max_x, min_y, max_y, seed):
    rng = np.random.default_rng(seed)  # Initialize the random number generator
    block_size = image.Y[min_y, min_x].size  # Determine the size of a block
    perm = rng.permutation(block_size)  # Generate a single permutation
    inv_perm = np.argsort(perm)  # Compute the inverse permutation

    for i in range(min_y, max_y):
        for j in range(min_x, max_x):
            block = image.Y[i, j]
            block = block.ravel()[inv_perm].reshape(block.shape)  # Apply the inverse permutation
            block = np.clip(block, -1023, 1023)  # Ensure coefficients are within valid range
            image.Y[i, j] = block

    return image


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


def scrambleface(img, first_frame, state):
    if state.scramble_type not in ["signFlip", "permutation"]:
        raise ValueError("Il tipo deve essere 'signFlip' o 'permutation'")

    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise ValueError("Impossibile leggere l'immagine")

    # Rileva i volti e raccogli le coordinate di ciascun volto
    landmarks_list = get_facial_landmarks(img)
    if not landmarks_list:
        return img  # Nessun volto rilevato

    data_str_list = []

    if first_frame and state.scramble_type == "signFlip":
        data_str_list.append(str(state.num_to_flip))

    for landmarks in landmarks_list:
        min_x = np.min(landmarks[:, 0]) // 8
        max_x = np.max(landmarks[:, 0]) // 8
        min_y = np.min(landmarks[:, 1]) // 8
        max_y = np.max(landmarks[:, 1]) // 8

        data_str = f"{min_x} {max_x} {min_y} {max_y}"
        data_str_list.append(data_str)

    concatenated_data_str = " ".join(data_str_list)
    data_bytes = concatenated_data_str.encode('utf-8')
    ciphertext = encrypt_string(state.key, data_bytes)
    encoded_ciphertext = rs.encode(ciphertext)

    # print("Length of the watermark:", len(encoded_ciphertext) * 8)

    # Salva temporaneamente l'immagine e leggi i coefficienti DCT
    for attempt in range(max_retries):
        success = cv2.imwrite('frame.jpg', img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
        if success:
            break
        time.sleep(retry_delay)
    image = jpeglib.read_dct('frame.jpg')

    # Applica le trasformazioni di scrambling nelle regioni facciali
    for landmarks in landmarks_list:
        min_x = np.min(landmarks[:, 0]) // 8
        max_x = np.max(landmarks[:, 0]) // 8
        min_y = np.min(landmarks[:, 1]) // 8
        max_y = np.max(landmarks[:, 1]) // 8

        if state.scramble_type == "signFlip":
            image = scramble_sign_flip(image, min_x, max_x, min_y, max_y, state.num_to_flip, state.seed)
        elif state.scramble_type == "permutation":
            image = scramble_permutation(image, min_x, max_x, min_y, max_y, state.seed)

    for attempt in range(max_retries):
        try:
            image.write_dct('output_scrambled.jpg')
            break
        except Exception as e:
            print(f"Errore nella scrittura del DCT (tentativo {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise e

    scrambled_image = cv2.imread('output_scrambled.jpg')

    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', encoded_ciphertext)
    img_encoded = encoder.encode(scrambled_image, 'dwtDctSvd')
    return img_encoded


def descrambleface(img, first_frame, state):
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise ValueError("Impossibile leggere l'immagine")

    initial_length = 376
    max_attempts = 5
    length_increment = 128

    for attempt in range(max_attempts):
        current_length = initial_length + attempt * length_increment
        decoder = WatermarkDecoder('bytes', current_length)
        watermark = decoder.decode(img, 'dwtDctSvd')

        ciphertext = watermark
        for retry in range(max_retries):
            success = cv2.imwrite('frame.jpg', img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
            if success:
                break
            time.sleep(retry_delay)

        try:
            ciphertext, _, errors = rs.decode(ciphertext)
        except ReedSolomonError:
            # print("Troppi errori di decodifica, riprova con un'altra lunghezza")
            continue
        extracted_data = decrypt_string(state.key, ciphertext)
        extracted_data = extracted_data.decode('utf-8')

        data_list = extracted_data.split()

        if first_frame:
            if len(data_list) % 2 == 1:
                state.scramble_type = "signFlip"
                state.num_to_flip = int(data_list[0])
                data_list = data_list[1:]
            else:
                state.scramble_type = "permutation"

        # Calculate the expected length based on the current watermark length
        expected_length = 4 + (current_length - initial_length) // length_increment * 4

        if len(data_list) == expected_length:
            # print(f"Lunghezza del watermark corretta estratta: {current_length}")
            break
    else:
        raise ValueError(
            "Errore nell'estrazione del watermark: numero di valori non corrispondente o nessun watermark nel frame")

    num_faces = len(data_list) // 4
    face_regions = []

    for i in range(num_faces):
        face_region = [int(data_list[j]) for j in range(i * 4, (i + 1) * 4)]
        face_regions.append(face_region)

    image = jpeglib.read_dct('frame.jpg')
    for face_region in face_regions:
        min_x, max_x, min_y, max_y = face_region
        if state.scramble_type == "signFlip":
            image = descramble_sign_flip(image, min_x, max_x, min_y, max_y, state.num_to_flip, state.seed)
        elif state.scramble_type == "permutation":
            image = descramble_permutation(image, min_x, max_x, min_y, max_y, state.seed)

    image.write_dct('output_descrambled.jpg')
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
                scrambled_frame = scrambleface(frame, first_frame, state)
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

    safe_remove('frame.jpg')
    safe_remove('output_scrambled.jpg')


def descramblevideo(input_video_path, output_video_path=None, key=None, progress_callback=None):
    if len(key) not in [16, 24, 32]:
        print("Chiave errata!")
        exit(1)

    key_hash = hashlib.sha256(key).hexdigest()
    seed = int(key_hash, 16)

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
                descrambled_frame = descrambleface(frame, first_frame, state)
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

    safe_remove('frame.jpg')
    safe_remove('output_descrambled.jpg')
