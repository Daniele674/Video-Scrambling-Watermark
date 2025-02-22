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
from concurrent.futures import ThreadPoolExecutor, as_completed


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

rs = RSCodec(15)  # 15 simboli per correzione (10 ecc)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()


def get_facial_landmarks(frame):
    """Rileva i landmark facciali nel frame."""
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
            matrix = np.clip(matrix, -1023, 1023)
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
            matrix = np.clip(matrix, -1023, 1023)
            image.Y[i, j] = matrix
    return image


def scramble_permutation(image, min_x, max_x, min_y, max_y, seed):
    rng = np.random.default_rng(seed)
    block_size = image.Y[min_y, min_x].size
    perm = rng.permutation(block_size)
    for i in range(min_y, max_y):
        for j in range(min_x, max_x):
            block = image.Y[i, j]
            block = block.ravel()[perm].reshape(block.shape)
            block = np.clip(block, -1023, 1023)
            image.Y[i, j] = block
    return image


def descramble_permutation(image, min_x, max_x, min_y, max_y, seed):
    rng = np.random.default_rng(seed)
    block_size = image.Y[min_y, min_x].size
    perm = rng.permutation(block_size)
    inv_perm = np.argsort(perm)
    for i in range(min_y, max_y):
        for j in range(min_x, max_x):
            block = image.Y[i, j]
            block = block.ravel()[inv_perm].reshape(block.shape)
            block = np.clip(block, -1023, 1023)
            image.Y[i, j] = block
    return image


def encrypt_string(key, plaintext):
    backend = default_backend()
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(128).padder()
    padded_plaintext = padder.update(plaintext) + padder.finalize()
    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
    return iv + ciphertext


def decrypt_string(key, ciphertext):
    backend = default_backend()
    iv = ciphertext[:16]
    ciphertext = ciphertext[16:]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
    decryptor = cipher.decryptor()
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
    return plaintext


def scrambleface(img, first_frame, state):
    """
    Funzione originale per applicare lo scrambling su un frame,
    che utilizza file temporanei a nome fisso.
    """
    if state.scramble_type not in ["signFlip", "permutation"]:
        raise ValueError("Il tipo deve essere 'signFlip' o 'permutation'")
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise ValueError("Impossibile leggere l'immagine")
    landmarks_list = get_facial_landmarks(img)
    data_str_list = []
    if first_frame:
        if state.scramble_type == "signFlip":
            data_str_list.append(str(state.num_to_flip))
        if not landmarks_list:
            data_str_list.append("0 0 0 0")
    if not landmarks_list:
        return img
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
    for attempt in range(max_retries):
        success = cv2.imwrite('frame.jpg', img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
        if success:
            break
        time.sleep(retry_delay)
    image = jpeglib.read_dct('frame.jpg')
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
    """
    Funzione originale per il descrambling di un frame.
    """
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
            continue
        extracted_data = decrypt_string(state.key, ciphertext)
        extracted_data = extracted_data.decode('utf-8')
        data_list = extracted_data.split()
        print('Data list:', data_list)
        if first_frame:
            if len(data_list) % 2 == 1:
                state.scramble_type = "signFlip"
                state.num_to_flip = int(data_list[0])
                data_list = data_list[1:]
            else:
                state.scramble_type = "permutation"
        expected_length = 4 + (current_length - initial_length) // length_increment * 4
        if len(data_list) == expected_length:
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


def safe_remove(file_path):
    """Elimina il file specificato se esiste."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Errore nella rimozione di {file_path}: {e}")


def create_video_with_ffmpeg(fps, output_video_path, output_data, extra_args=None):
    command = [
        'ffmpeg',
        '-y',
        '-r', str(fps),
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-i', '-',
        '-vcodec', 'libx264'
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


# --- Versione Parallelizzata delle funzioni di scrambling/descrambling ---

def scrambleface_parallel(img, first_frame, state, temp_prefix):
    """
    Versione modificata di scrambleface che utilizza file temporanei univoci.
    I nomi dei file sono basati su temp_prefix per evitare conflitti in elaborazioni parallele.
    """
    landmarks_list = get_facial_landmarks(img)
    data_str_list = []
    if first_frame:
        if state.scramble_type == "signFlip":
            data_str_list.append(str(state.num_to_flip))
        if not landmarks_list:
            data_str_list.append("0 0 0 0")
    if not landmarks_list:
        return img
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
    temp_frame = f'{temp_prefix}_frame.jpg'
    temp_out = f'{temp_prefix}_output_scrambled.jpg'
    for attempt in range(max_retries):
        success = cv2.imwrite(temp_frame, img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
        if success:
            break
        time.sleep(retry_delay)
    image = jpeglib.read_dct(temp_frame)
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
            image.write_dct(temp_out)
            break
        except Exception as e:
            print(f"Errore nella scrittura del DCT (tentativo {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise e
    scrambled_image = cv2.imread(temp_out)
    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', encoded_ciphertext)
    img_encoded = encoder.encode(scrambled_image, 'dwtDctSvd')
    return img_encoded


def scramblevideo_parallel(input_video_path, output_video_path, scramble_settings=None, key=None,
                           progress_callback=None):
    """
    Elabora i frame in parallelo e li riordina per creare il video finale.
    """
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
    frames_data = []
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        frames_data.append((frame_number, frame))
    cap.release()
    state = ScrambleState(
        seed=seed,
        scramble_type=scramble_settings['scramble_type'],
        num_to_flip=scramble_settings.get('num_to_flip', 0),
        key=key
    )
    results = {}

    def process_frame(frame_info):
        idx, frame = frame_info
        first_frame = (idx == 1)
        temp_prefix = f"temp_{idx}"
        try:
            processed_frame = scrambleface_parallel(frame, first_frame, state, temp_prefix)
        except Exception as e:
            print(f"Errore nell'elaborazione del frame {idx}: {str(e)}")
            processed_frame = frame
        success, buffer = cv2.imencode('.jpg', processed_frame)
        if not success:
            print(f"Errore nella codifica del frame {idx}")
            return idx, None
        safe_remove(f'{temp_prefix}_frame.jpg')
        safe_remove(f'{temp_prefix}_output_scrambled.jpg')
        return idx, buffer

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {executor.submit(process_frame, fd): fd[0] for fd in frames_data}
        for future in as_completed(future_to_idx):
            idx, buffer = future.result()
            results[idx] = buffer
            if progress_callback:
                progress_callback(idx, total_frames)
    ordered_frames = [results[i] for i in sorted(results.keys()) if results[i] is not None]
    with io.BytesIO() as output:
        for frame in ordered_frames:
            output.write(frame.tobytes())
        output_data = output.getvalue()
    extra_args = ['-preset', 'slow', '-q:v', '0', '-crf', '10']
    create_video_with_ffmpeg(fps, output_video_path, output_data, extra_args)
    print("Elaborazione video completata.")


def descrambleface_parallel(img, first_frame, state, temp_prefix):
    """
    Versione modificata di descrambleface per l'elaborazione parallela.
    Utilizza file temporanei univoci basati su temp_prefix.
    """
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
        temp_frame = f'{temp_prefix}_frame.jpg'
        for retry in range(max_retries):
            success = cv2.imwrite(temp_frame, img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
            if success:
                break
            time.sleep(retry_delay)
        try:
            ciphertext, _, errors = rs.decode(ciphertext)
        except ReedSolomonError:
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
        expected_length = 4 + (current_length - initial_length) // length_increment * 4
        if len(data_list) == expected_length:
            break
    else:
        raise ValueError(
            "Errore nell'estrazione del watermark: numero di valori non corrispondente o nessun watermark nel frame")
    num_faces = len(data_list) // 4
    face_regions = []
    for i in range(num_faces):
        face_region = [int(data_list[j]) for j in range(i * 4, (i + 1) * 4)]
        face_regions.append(face_region)
    image = jpeglib.read_dct(temp_frame)
    for face_region in face_regions:
        min_x, max_x, min_y, max_y = face_region
        if state.scramble_type == "signFlip":
            image = descramble_sign_flip(image, min_x, max_x, min_y, max_y, state.num_to_flip, state.seed)
        elif state.scramble_type == "permutation":
            image = descramble_permutation(image, min_x, max_x, min_y, max_y, state.seed)
    temp_out = f'{temp_prefix}_output_descrambled.jpg'
    image.write_dct(temp_out)
    descrambled_image = cv2.imread(temp_out)
    safe_remove(temp_frame)
    safe_remove(temp_out)
    return descrambled_image


def descramblevideo_parallel(input_video_path, output_video_path, key=None, progress_callback=None):
    """
    Versione parallelizzata della funzione di descrambling video.
    """
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
    frames_data = []
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        frames_data.append((frame_number, frame))
    cap.release()
    state = ScrambleState(seed=seed, key=key)
    results = {}

    def process_frame(frame_info):
        idx, frame = frame_info
        first_frame = (idx == 1)
        temp_prefix = f"temp_{idx}"
        try:
            processed_frame = descrambleface_parallel(frame, first_frame, state, temp_prefix)
        except Exception as e:
            print(f"Errore nell'elaborazione del frame {idx}: {str(e)}")
            processed_frame = frame
        success, buffer = cv2.imencode('.jpg', processed_frame)
        if not success:
            print(f"Errore nella codifica del frame {idx}")
            return idx, None
        return idx, buffer

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {executor.submit(process_frame, fd): fd[0] for fd in frames_data}
        for future in as_completed(future_to_idx):
            idx, buffer = future.result()
            results[idx] = buffer
            if progress_callback:
                progress_callback(idx, total_frames)
    ordered_frames = [results[i] for i in sorted(results.keys()) if results[i] is not None]
    with io.BytesIO() as output:
        for frame in ordered_frames:
            output.write(frame.tobytes())
        output_data = output.getvalue()
    extra_args = ['-crf', '23']
    create_video_with_ffmpeg(fps, output_video_path, output_data, extra_args)
    print("Elaborazione video completata.")

# Esempio di utilizzo:
# scramblevideo_parallel("input_video.mp4", "output_scrambled.mp4", scramble_settings={'scramble_type': 'signFlip', 'num_to_flip': 5}, key=b'questa_e_una_chiave_16')
# descramblevideo_parallel("output_scrambled.mp4", "output_descrambled.mp4", key=b'questa_e_una_chiave_16')
