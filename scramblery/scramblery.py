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
import tempfile
import multiprocessing as mproc  # alias diverso per multiprocessing

# Classe per incapsulare lo stato di scrambling/descrambling
class ScrambleState:
    def __init__(self, seed=None, scramble_type=None, num_to_flip=None, face_region=None, key=None):
        self.seed = seed
        self.scramble_type = scramble_type
        self.num_to_flip = num_to_flip
        self.face_region = face_region
        self.key = key

max_retries = 3  # Numero massimo di tentativi
retry_delay = 0.5  # Secondi di attesa tra un tentativo

rs = RSCodec(15)  # 10 ecc symbols

# Funzione initializer per ogni worker: ogni processo crea la propria istanza di FaceMesh
def init_worker():
    global face_mesh_detector
    face_mesh_detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

def get_facial_landmarks(frame):
    """Rileva i landmark facciali utilizzando l'istanza globale."""
    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh_detector.process(frame_rgb)
    facelandmarks_list = []
    if result.multi_face_landmarks:
        for facial_landmarks in result.multi_face_landmarks:
            pts = np.array([[int(pt.x * width), int(pt.y * height)]
                             for pt in facial_landmarks.landmark[:468]], np.int32)
            facelandmarks_list.append(pts)
    return facelandmarks_list

def sign_flip(image, min_x, max_x, min_y, max_y, num_to_flip, seed):
    """Applica la trasformazione 'sign flip' su una regione."""
    rng = np.random.default_rng(seed)
    Y = image.Y
    for i in range(min_y, max_y):
        for j in range(min_x, max_x):
            matrix = Y[i, j]
            shape = matrix.shape
            indices = rng.choice(np.prod(shape), num_to_flip)
            coords = np.unravel_index(indices, shape)
            matrix[coords] *= -1
            Y[i, j] = np.clip(matrix, -1023, 1023)
    return image

def scramble_permutation(image, min_x, max_x, min_y, max_y, seed):
    rng = np.random.default_rng(seed)
    Y = image.Y
    block_size = Y[min_y, min_x].size
    perm = rng.permutation(block_size)
    for i in range(min_y, max_y):
        for j in range(min_x, max_x):
            block = Y[i, j]
            block = block.ravel()[perm].reshape(block.shape)
            Y[i, j] = np.clip(block, -1023, 1023)
    return image

def descramble_permutation(image, min_x, max_x, min_y, max_y, seed):
    rng = np.random.default_rng(seed)
    Y = image.Y
    block_size = Y[min_y, min_x].size
    perm = rng.permutation(block_size)
    inv_perm = np.argsort(perm)
    for i in range(min_y, max_y):
        for j in range(min_x, max_x):
            block = Y[i, j]
            block = block.ravel()[inv_perm].reshape(block.shape)
            Y[i, j] = np.clip(block, -1023, 1023)
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
    if state.scramble_type not in ["signFlip", "permutation"]:
        raise ValueError("Il tipo deve essere 'signFlip' o 'permutation'")

    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise ValueError("Impossibile leggere l'immagine")

    # Rileva i volti e ottieni le coordinate
    landmarks_list = get_facial_landmarks(img)
    region_coords = []
    data_str_list = []

    if first_frame:
        if state.scramble_type == "signFlip":
            data_str_list.append(str(state.num_to_flip))
        # Se non viene rilevato alcun volto nel primo frame, aggiungi placeholder
        if not landmarks_list:
            data_str_list.extend(["0", "0", "0", "0"])
            concatenated_data_str = " ".join(data_str_list)
            data_bytes = concatenated_data_str.encode('utf-8')
            ciphertext = encrypt_string(state.key, data_bytes)
            encoded_ciphertext = rs.encode(ciphertext)
            encoder = WatermarkEncoder()
            encoder.set_watermark('bytes', encoded_ciphertext)
            img_encoded = encoder.encode(img, 'dwtDctSvd')
            return img_encoded

    for landmarks in landmarks_list:
        min_x = np.min(landmarks[:, 0]) // 8
        max_x = np.max(landmarks[:, 0]) // 8
        min_y = np.min(landmarks[:, 1]) // 8
        max_y = np.max(landmarks[:, 1]) // 8
        region_coords.append((min_x, max_x, min_y, max_y))
        data_str_list.append(f"{min_x} {max_x} {min_y} {max_y}")

    concatenated_data_str = " ".join(data_str_list)
    data_bytes = concatenated_data_str.encode('utf-8')
    ciphertext = encrypt_string(state.key, data_bytes)
    encoded_ciphertext = rs.encode(ciphertext)

    # Scrittura temporanea per generare la DCT
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_frame_path = tmp_file.name
    for attempt in range(max_retries):
        success = cv2.imwrite(temp_frame_path, img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
        if success:
            break
        time.sleep(retry_delay)
    image = jpeglib.read_dct(temp_frame_path)

    # Applica le trasformazioni nelle regioni facciali rilevate
    for (min_x, max_x, min_y, max_y) in region_coords:
        if state.scramble_type == "signFlip":
            image = sign_flip(image, min_x, max_x, min_y, max_y, state.num_to_flip, state.seed)
        elif state.scramble_type == "permutation":
            image = scramble_permutation(image, min_x, max_x, min_y, max_y, state.seed)

    # Salva su file temporaneo per la scrittura DCT
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_out:
        temp_out_path = tmp_out.name
    for attempt in range(max_retries):
        try:
            image.write_dct(temp_out_path)
            break
        except Exception as e:
            print(f"Errore nella scrittura del DCT (tentativo {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise e

    scrambled_image = cv2.imread(temp_out_path)
    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', encoded_ciphertext)
    img_encoded = encoder.encode(scrambled_image, 'dwtDctSvd')

    safe_remove(temp_frame_path)
    safe_remove(temp_out_path)
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
        # Utilizza un file temporaneo per la lettura JPEG
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            temp_frame_path = tmp_file.name
        for retry in range(max_retries):
            success = cv2.imwrite(temp_frame_path, img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
            if success:
                break
            time.sleep(retry_delay)
        try:
            ciphertext, _, errors = rs.decode(ciphertext)
        except ReedSolomonError:
            safe_remove(temp_frame_path)
            continue
        extracted_data = decrypt_string(state.key, ciphertext)
        extracted_data = extracted_data.decode('utf-8')
        data_list = extracted_data.split()

        if first_frame:
            # Controlla se il watermark corrisponde al placeholder
            if len(data_list) == 5 and data_list[1:] == ["0", "0", "0", "0"]:
                state.scramble_type = "signFlip"
                state.num_to_flip = int(data_list[0])
                return img
            elif len(data_list) == 4 and data_list == ["0", "0", "0", "0"]:
                state.scramble_type = "permutation"
                return img
            elif len(data_list) % 2 == 1:
                state.scramble_type = "signFlip"
                state.num_to_flip = int(data_list[0])
                data_list = data_list[1:]
            else:
                state.scramble_type = "permutation"

        expected_length = 4 + (current_length - initial_length) // length_increment * 4
        if len(data_list) == expected_length:
            break
    else:
        safe_remove(temp_frame_path)
        raise ValueError("Errore nell'estrazione del watermark: numero di valori non corrispondente o nessun watermark nel frame")

    num_faces = len(data_list) // 4
    face_regions = []
    for i in range(num_faces):
        region = [int(data_list[j]) for j in range(i * 4, (i + 1) * 4)]
        face_regions.append(region)

    image = jpeglib.read_dct(temp_frame_path)
    safe_remove(temp_frame_path)
    for face_region in face_regions:
        min_x, max_x, min_y, max_y = face_region
        if state.scramble_type == "signFlip":
            image = sign_flip(image, min_x, max_x, min_y, max_y, state.num_to_flip, state.seed)
        elif state.scramble_type == "permutation":
            image = descramble_permutation(image, min_x, max_x, min_y, max_y, state.seed)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_out:
        temp_out_path = tmp_out.name
    image.write_dct(temp_out_path)
    descrambled_image = cv2.imread(temp_out_path)
    safe_remove(temp_out_path)
    return descrambled_image


def safe_remove(file_path):
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

# --- FUNZIONI DI MULTIPROCESSING PER ELABORARE UN FRAME ---
def process_frame_scramble(args):
    frame, first_frame, state = args
    try:
        processed_frame = scrambleface(frame, first_frame, state)
    except Exception as e:
        print(f"Errore nell'elaborazione del frame: {e}")
        processed_frame = frame
    return processed_frame

def process_frame_descramble(args):
    frame, first_frame, state = args
    try:
        processed_frame = descrambleface(frame, first_frame, state)
    except Exception as e:
        print(f"Errore nell'elaborazione del frame: {e}")
        processed_frame = frame
    return processed_frame

# Funzione per elaborare video in modalità "scramble" usando multiprocessing
def scramblevideo(input_video_path, output_video_path=None, scramble_settings=None, key=None, progress_callback=None):
    if scramble_settings is None:
        scramble_settings = {'scramble_type': 'permutation', 'num_to_flip': 0}
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
    state = ScrambleState(seed=seed, scramble_type=scramble_settings['scramble_type'],
                          num_to_flip=scramble_settings.get('num_to_flip', 0), key=key)
    audio_output_path = 'extracted_audio.aac'
    has_audio = extract_audio(input_video_path, audio_output_path)
    frame_args = []
    with tqdm(total=total_frames, desc="Elaborazione frame", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            if progress_callback:
                progress_callback(frame_number, total_frames)
            first_frame = (frame_number == 1)
            frame_args.append((frame, first_frame, state))
            pbar.update(1)
    cap.release()
    # Crea un pool di processi (inizializza ogni worker con init_worker)
    pool = mproc.Pool(processes=mproc.cpu_count(), initializer=init_worker)
    processed_frames = pool.map(process_frame_scramble, frame_args)
    pool.close()
    pool.join()
    cv2.destroyAllWindows()
    print("Elaborazione video completata.")
    with io.BytesIO() as output:
        for frame in processed_frames:
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                output.write(buffer.tobytes())
            else:
                print("Errore nella codifica di un frame")
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

# Funzione per elaborare video in modalità "descramble" usando multiprocessing
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
    state = ScrambleState(seed=seed, key=key)

    # Estrai il primo frame e usalo per aggiornare lo stato (metadati per il descrambling)
    ret, first_frame = cap.read()
    if not ret:
        print("Errore nella lettura del primo frame")
        return
    # Elaboriamo il primo frame in modo sequenziale per estrarre le informazioni
    # (first_frame=True)
    try:
        processed_first = descrambleface(first_frame, True, state)
    except Exception as e:
        print(f"Errore nell'elaborazione del primo frame: {e}")
        processed_first = first_frame

    frame_args = []
    frame_number = 1  # il primo frame è già stato processato
    with tqdm(total=total_frames - 1, desc="Elaborazione frame", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            if progress_callback:
                progress_callback(frame_number, total_frames)
            # Imposta first_frame=False per tutti gli altri frame,
            # poiché lo stato è già aggiornato dal primo frame.
            frame_args.append((frame, False, state))
            pbar.update(1)
    cap.release()

    # Processa in parallelo i frame rimanenti
    pool = mproc.Pool(processes=mproc.cpu_count(), initializer=init_worker)
    processed_frames = pool.map(process_frame_descramble, frame_args)
    pool.close()
    pool.join()
    cv2.destroyAllWindows()
    print("Elaborazione video completata.")

    # Prependi il primo frame processato all'elenco dei frame
    all_frames = [processed_first] + processed_frames

    with io.BytesIO() as output:
        for frame in all_frames:
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                output.write(buffer.tobytes())
            else:
                print("Errore nella codifica di un frame")
        output_data = output.getvalue()
    extra_args = ['-crf', '23']
    create_video_with_ffmpeg(fps, output_video_path, output_data, extra_args)
    # Se presente, ricomponi anche l'audio
    audio_output_path = 'extracted_audio.aac'
    if extract_audio(input_video_path, audio_output_path):
        final_output_path = 'temp_output_video.mp4'
        combine_audio_video(output_video_path, audio_output_path, final_output_path)
        safe_remove(output_video_path)
        os.rename(final_output_path, output_video_path)
        safe_remove(audio_output_path)
    safe_remove('frame.jpg')
    safe_remove('output_descrambled.jpg')
