import sys

import mediapipe as mp
import cv2
import numpy as np
import os
import random
from blind_watermark import WaterMark
import jpeglib
from imwatermark import WatermarkEncoder, WatermarkDecoder
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
import random


def get_facial_landmarks(frame):
    """a function for detecting facial landmarks"""
    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    facelandmarks = []
    for facial_landmarks in result.multi_face_landmarks:
        for i in range(0, 468):
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
            facelandmarks.append([x, y])
    return np.array(facelandmarks, np.int32)


# permutazione solo coefficienti AC
# def scramble_dct_block(block, seed):
#     # Flatten the block into a 1D array
#     flat_block = block.flatten()
#
#     # Exclude the first element (DC coefficient) during permutation
#     ac_coefficients = flat_block[1:]
#
#     # Generate a permutation of the indices
#     np.random.seed(seed)
#     perm = np.random.permutation(len(ac_coefficients))
#
#     # Apply the permutation to the AC coefficients
#     scrambled_ac_coefficients = ac_coefficients[perm]
#
#     # Combine the DC coefficient with the scrambled AC coefficients
#     scrambled_block = np.concatenate(([flat_block[0]], scrambled_ac_coefficients))
#
#     # Reshape the scrambled block back into the original shape
#     return scrambled_block.reshape(block.shape)


# def descramble_dct_block(block, seed):
#     # Flatten the block into a 1D array
#     flat_block = block.flatten()
#
#     # Exclude the first element (DC coefficient) during permutation
#     ac_coefficients = flat_block[1:]
#
#     # Generate the original permutation of the indices
#     np.random.seed(seed)
#     perm = np.random.permutation(len(ac_coefficients))
#
#     # Generate the inverse permutation
#     inv_perm = np.argsort(perm)
#
#     # Apply the inverse permutation to the AC coefficients
#     descrambled_ac_coefficients = ac_coefficients[inv_perm]
#
#     # Combine the DC coefficient with the descrambled AC coefficients
#     descrambled_block = np.concatenate(([flat_block[0]], descrambled_ac_coefficients))
#
#     # Reshape the descrambled block back into the original shape
#     return descrambled_block.reshape(block.shape)

# permutazione anche del coefficiente DC
def scramble_dct_block(block, seed):
    # Flatten the block into a 1D array
    flat_block = block.flatten()

    # Generate a permutation of the indices
    np.random.seed(seed)
    perm = np.random.permutation(len(flat_block))

    # Apply the permutation to the coefficients
    scrambled_coefficients = flat_block[perm]

    # Reshape the scrambled block back into the original shape
    return scrambled_coefficients.reshape(block.shape)


def descramble_dct_block(block, seed):
    # Flatten the block into a 1D array
    flat_block = block.flatten()

    # Generate the original permutation of the indices
    np.random.seed(seed)
    perm = np.random.permutation(len(flat_block))

    # Generate the inverse permutation
    inv_perm = np.argsort(perm)

    # Apply the inverse permutation to the coefficients
    descrambled_coefficients = flat_block[inv_perm]

    # Reshape the descrambled block back into the original shape
    return descrambled_coefficients.reshape(block.shape)


# Encryption and Decryption functions
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


def descrambleface(img, previous_watermark, wm_shape, password_img, password_wm, seed=None, write=True):
    img_name = None
    extracted_data = ""

    if isinstance(img, str):
        image_path = img
        img = cv2.imread(img)
        if img is None:
            raise ValueError("Could not read the image")

        img_name = os.path.splitext(os.path.basename(image_path))[0]

    # cv2.imwrite('frame.jpg', img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
    # bwm1 = WaterMark(password_img=password_img, password_wm=password_wm)
    # extracted_data = bwm1.extract(filename='frame.jpg', wm_shape=wm_shape, mode='str')
    try:
        key = b'supersegreto1234'  # Chiave di decifratura

        decoder = WatermarkDecoder('bytes', 384)
        # Decode the watermark from the image
        watermark = decoder.decode(img, 'dwtDctSvd')
        # The watermark is in bytes, so we decode it to string
        ciphertext = watermark

        extracted_data = decrypt_string(key, ciphertext)

        print(extracted_data)

    except Exception as e:
        print(f"An error occurred during watermark extraction: {str(e)}")
        print("Using the previous watermark.")
        extracted_data = previous_watermark

    cv2.imwrite('frame.jpg', img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])

    # Split the extracted_data string into a list of substrings
    data_list = extracted_data.split()

    # Assign the first string to seed_number
    try:
        seed_number = int(data_list[0])
    except ValueError:
        print("Invalid seed number. I use:", previous_watermark.split()[0])
        seed_number = int(previous_watermark.split()[0])

    # Assign the last four strings to face_region
    try:
        face_region = [int(i) for i in data_list[1:5]]
    except ValueError:
        print("Invalid face region values. I use:", previous_watermark.split()[1:5])
        face_region = [int(i) for i in previous_watermark.split()[1:5]]

    type = data_list[5]

    # Extract num_to_flip from the watermark
    if type == "signFlip":
        num_to_flip = int(data_list[6])

    np.random.seed(seed_number)

    min_x, max_x, min_y, max_y = face_region

    if type == "signFlip":

        # Read DCT coefficients from the scrambled JPEG file
        scrambled_image = jpeglib.read_dct('frame.jpg')

        for i in range(min_y, max_y):
            for j in range(min_x, max_x):
                matrix = scrambled_image.Y[i, j]

                # Get the shape of the matrix
                shape = matrix.shape

                # Generate random indices
                indices = np.random.choice(np.prod(shape), num_to_flip)

                # Convert the indices to 2D coordinates
                coords = np.unravel_index(indices, shape)

                # Flip the sign of the selected coefficients
                matrix[coords] *= -1

                scrambled_image.Y[i, j] = matrix

        # Write descrambled coefficients back to a JPEG file
        scrambled_image.write_dct('output_descrambled.jpg')

        descrambled_image = cv2.imread('output_descrambled.jpg')

        return descrambled_image, extracted_data

    elif type == "permutation":

        # Read DCT coefficients from the scrambled JPEG file
        scrambled_image = jpeglib.read_dct('frame.jpg')

        # Seconda metodologia tramite permutazione dei coefficienti AC
        for i in range(min_y, max_y):
            for j in range(min_x, max_x):
                scrambled_image.Y[i, j] = descramble_dct_block(scrambled_image.Y[i, j], seed_number)

        # Write descrambled coefficients back to a JPEG file
        scrambled_image.write_dct('output_descrambled.jpg')

        descrambled_image = cv2.imread('output_descrambled.jpg')

        return descrambled_image, extracted_data


def scrambleface(img, type, password_img, password_wm, seed=None, write=True, num_to_flip=0):
    img_name = None

    if seed is not None:
        np.random.seed(seed)

    if type not in ["signFlip", "permutation"]:
        raise ValueError("type must be 'signFlip' or 'permutation'")

    if isinstance(img, str):
        image_path = img
        img = cv2.imread(img)
        if img is None:
            raise ValueError("Could not read the image")

        img_name = os.path.splitext(os.path.basename(image_path))[0]

    landmarks = get_facial_landmarks(img)
    # Define the bounding box around the face using the facial landmarks
    min_x = np.min(landmarks[:, 0]) // 8  # Dividing by 8 to match DCT block size
    max_x = np.max(landmarks[:, 0]) // 8
    min_y = np.min(landmarks[:, 1]) // 8
    max_y = np.max(landmarks[:, 1]) // 8

    data_str = str(seed) + ' ' + str(min_x) + ' ' + str(max_x) + ' ' + str(min_y) + ' ' + str(max_y) + ' ' + type

    if type == "signFlip":
        data_str = data_str + ' ' + str(num_to_flip)

    # bwm1 = WaterMark(password_img=password_img, password_wm=password_wm)
    # bwm1.read_img(img=img)
    # bwm1.read_wm(data_str, mode='str')
    # bwm1.embed('frame.jpg', compression_ratio=100)
    # len_wm = len(bwm1.wm_bit)
    # print('Put down the length of wm_bit {len_wm}'.format(len_wm=len_wm))

    key = b'supersegreto1234'  # Chiave di cifratura
    ciphertext = encrypt_string(key, data_str)

    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', ciphertext)
    img_encoded = encoder.encode(img, 'dwtDctSvd')
    cv2.imwrite('frame.jpg', img_encoded, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
    # print("Length of the watermark:", len(ciphertext) * 8)

    if type == "signFlip":

        image = jpeglib.read_dct('frame.jpg')

        for i in range(min_y, max_y):
            for j in range(min_x, max_x):
                matrix = image.Y[i, j]

                # Get the shape of the matrix
                shape = matrix.shape

                # Generate random indices
                indices = np.random.choice(np.prod(shape), num_to_flip)

                # Convert the indices to 2D coordinates
                coords = np.unravel_index(indices, shape)

                # Flip the sign of the selected coefficients
                matrix[coords] *= -1

                image.Y[i, j] = matrix

        image.write_dct('output_scrambled.jpg')

        scrambled_image = cv2.imread('output_scrambled.jpg')

        return scrambled_image

    elif type == "permutation":

        image = jpeglib.read_dct('frame.jpg')

        # Seconda metodologia tramite permutazione
        for i in range(min_y, max_y):
            for j in range(min_x, max_x):
                image.Y[i, j] = scramble_dct_block(image.Y[i, j], seed)

        image.write_dct('output_scrambled.jpg')

        scrambled_image = cv2.imread('output_scrambled.jpg')

        return scrambled_image


def update_progress(current_frame, total_frames):
    # Calcola la percentuale di completamento
    progress_percentage = (current_frame / total_frames) * 100

    # Aggiorna la barra di avanzamento
    progress['value'] = progress_percentage
    root.update_idletasks()


# def scramblevideo(input_video_path, output_video_path=None, scramble_settings=None, progress_callback=None):
#     # Check if scramble_settings is provided, else use default settings
#     if scramble_settings is None:
#         scramble_settings = {
#             'type': 'permutation',
#             'num_to_flip': 0,  # Number of coefficients to flip for 'signFlip'
#             'seed': 1,
#             'password_img': 1,
#             'password_wm': 1,
#             'write': False  # Important: this should always be False for video processing
#         }
#     else:
#         if "write" in scramble_settings and scramble_settings["write"]:
#             print(
#                 "Warning: The 'write' setting in scramble_settings must be False for video processing. Overriding it to False.")
#             scramble_settings["write"] = False
#
#     # Open the video file
#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video {input_video_path}.")
#         return
#
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     # codec = cv2.VideoWriter_fourcc(*'XVID')
#     codec = cv2.VideoWriter_fourcc(*'MPEG')
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     if output_video_path:
#         out = cv2.VideoWriter(output_video_path, codec, fps, (frame_width, frame_height))
#     else:
#         out = None
#         print("No output path specified. The scrambled video won't be saved.")
#
#     frame_number = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Reached the end of the video.")
#             break
#
#         frame_number += 1
#         if progress_callback:
#             progress_callback(frame_number, total_frames)
#         print(f"Processing frame {frame_number}/{total_frames}")
#
#         try:
#             scrambled_frame = scrambleface(frame, **scramble_settings)
#         except Exception as e:
#             print(f"An error occurred while processing frame {frame_number}: {str(e)}")
#             scrambled_frame = frame
#
#         if out:
#             out.write(scrambled_frame)
#
#     cap.release()
#     if out:
#         out.release()
#
#     cv2.destroyAllWindows()
#     print("Video processing completed.")
#
#
# def descramblevideo(input_video_path, output_video_path=None, descramble_settings=None, progress_callback=None):
#     # Check if scramble_settings is provided, else use default settings
#     if descramble_settings is None:
#         descramble_settings = {
#             'seed': 1,
#             'wm_shape': None,
#             'password_img': 1,
#             'password_wm': 1,
#             'write': False  # Important: this should always be False for video processing
#         }
#     else:
#         if "write" in descramble_settings and descramble_settings["write"]:
#             print(
#                 "Warning: The 'write' setting in scramble_settings must be False for video processing. Overriding it to False.")
#             descramble_settings["write"] = False
#
#     # Open the video file
#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video {input_video_path}.")
#         return
#
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     # codec = cv2.VideoWriter_fourcc(*'XVID')
#     codec = cv2.VideoWriter_fourcc(*'MPEG')
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     if output_video_path:
#         out = cv2.VideoWriter(output_video_path, codec, fps, (frame_width, frame_height))
#     else:
#         out = None
#         print("No output path specified. The scrambled video won't be saved.")
#
#     frame_number = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Reached the end of the video.")
#             break
#
#         frame_number += 1
#         if progress_callback:
#             progress_callback(frame_number, total_frames)
#         print(f"Processing frame {frame_number}/{total_frames}")
#
#         try:
#             descrambled_frame = descrambleface(frame, **descramble_settings)
#         except Exception as e:
#             print(f"An error occurred while processing frame {frame_number}: {str(e)}")
#             descrambled_frame = frame
#
#         if out:
#             out.write(descrambled_frame)
#
#     cap.release()
#     if out:
#         out.release()
#
#     cv2.destroyAllWindows()
#     print("Video processing completed.")

import cv2
import subprocess
import os


def scramblevideo(input_video_path, output_video_path=None, scramble_settings=None, progress_callback=None):
    # Check if scramble_settings is provided, else use default settings
    if scramble_settings is None:
        scramble_settings = {
            'type': 'permutation',
            'num_to_flip': 0,  # Number of coefficients to flip for 'signFlip'
            'seed': 1,
            'password_img': 1,
            'password_wm': 1,
            'write': False  # Important: this should always be False for video processing
        }
    else:
        if "write" in scramble_settings and scramble_settings["write"]:
            print(
                "Warning: The 'write' setting in scramble_settings must be False for video processing. Overriding it to False.")
            scramble_settings["write"] = False

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Reached the end of the video.")
            break

        frame_number += 1
        if progress_callback:
            progress_callback(frame_number, total_frames)
        print(f"Processing frame {frame_number}/{total_frames}")

        try:
            scrambled_frame = scrambleface(frame, **scramble_settings)
        except Exception as e:
            print(f"An error occurred while processing frame {frame_number}: {str(e)}")
            scrambled_frame = frame

        # Write each frame to a temporary image file
        cv2.imwrite(f"temp/frame_{frame_number:04d}.jpg", scrambled_frame)

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")

    # Use ffmpeg to convert the image sequence to a video file
    compression_ratio = 10  # Adjust this value to control the compression ratio
    subprocess.call(
        f"ffmpeg -y -r {fps} -i temp/frame_%04d.jpg -vcodec libx264 -crf {compression_ratio} {output_video_path}",
        shell=True)
    # subprocess.call(f"ffmpeg -r {fps} -i temp/frame_%04d.jpg -vcodec mpeg1video -q:v 1 {output_video_path}", shell=True)

    # Delete temporary image files
    for file_name in os.listdir("temp"):
        if file_name.endswith(".jpg"):
            os.remove(f"temp/{file_name}")


def descramblevideo(input_video_path, output_video_path=None, descramble_settings=None, progress_callback=None):
    # Check if scramble_settings is provided, else use default settings
    if descramble_settings is None:
        descramble_settings = {
            'seed': 1,
            'wm_shape': None,
            'password_img': 1,
            'password_wm': 1,
            'write': False  # Important: this should always be False for video processing
        }
    else:
        if "write" in descramble_settings and descramble_settings["write"]:
            print(
                "Warning: The 'write' setting in scramble_settings must be False for video processing. Overriding it to False.")
            descramble_settings["write"] = False

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    previous_watermark = ""

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Reached the end of the video.")
            break

        frame_number += 1
        if progress_callback:
            progress_callback(frame_number, total_frames)
        print(f"Processing frame {frame_number}/{total_frames}")

        try:
            descrambled_frame, current_watermark = descrambleface(frame, previous_watermark, **descramble_settings)
            previous_watermark = current_watermark
        except Exception as e:
            print(f"An error occurred while processing frame {frame_number}: {str(e)}")
            descrambled_frame = frame

        # Write each frame to a temporary image file
        cv2.imwrite(f"temp/frame_{frame_number:04d}.jpg", descrambled_frame)

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")

    # Use ffmpeg to convert the image sequence to a video file
    compression_ratio = 23  # Adjust this value to control the compression ratio
    subprocess.call(
        f"ffmpeg -y -r {fps} -i temp/frame_%04d.jpg -vcodec libx264 -crf {compression_ratio} {output_video_path}",
        shell=True)
    # subprocess.call(f"ffmpeg -r {fps} -i temp/frame_%04d.jpg -vcodec mpeg1video -q:v 16 {output_video_path}", shell=True)

    # Delete temporary image files
    for file_name in os.listdir("temp"):
        if file_name.endswith(".jpg"):
            os.remove(f"temp/{file_name}")
