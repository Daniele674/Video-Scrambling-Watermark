import jpeglib
import numpy as np
import cv2
import mediapipe as mp
from blind_watermark import WaterMark


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


seed_number = 1234

face = cv2.imread('testSet/persona.jpg')

np.random.seed(seed_number)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

facial_landmarks = get_facial_landmarks(face)

# Define the bounding box around the face using the facial landmarks
min_x = np.min(facial_landmarks[:, 0]) // 8  # Dividing by 8 to match DCT block size
max_x = np.max(facial_landmarks[:, 0]) // 8
min_y = np.min(facial_landmarks[:, 1]) // 8
max_y = np.max(facial_landmarks[:, 1]) // 8

data_str = str(seed_number) + ' ' + str(min_x) + ' ' + str(max_x) + ' ' + str(min_y) + ' ' + str(max_y)

print(data_str)

bwm1 = WaterMark(password_img=1, password_wm=1)
bwm1.read_img('testSet/persona.jpg')
bwm1.read_wm(data_str, mode='str')
bwm1.embed('persona.jpg', compression_ratio=100)
len_wm = len(bwm1.wm_bit)
print('Put down the length of wm_bit {len_wm}'.format(len_wm=len_wm))

print(min_x, max_x, min_y, max_y)

# Read DCT coefficients from a JPEG file
image = jpeglib.read_dct('persona.jpg')

# Scegli quanti coefficienti DCT cambiare di segno
num_to_flip = 60

print(image.Y[min_y:max_y, min_x:max_x][0])

# for i in range(min_y, max_y):
#     for j in range(min_x, max_x):
#         matrix = image.Y[i, j]
#
#         # Get the shape of the matrix
#         shape = matrix.shape
#
#         # Generate random indices
#         indices = np.random.choice(np.prod(shape), num_to_flip)
#
#         # Convert the indices to 2D coordinates
#         coords = np.unravel_index(indices, shape)
#
#         # Flip the sign of the selected coefficients
#         matrix[coords] *= -1
#
#         image.Y[i, j] = matrix

print('-' * 100)
print(image.Y[min_y:max_y, min_x:max_x][0])


# scrambling non soddisfacente
# for i in range(min_y, max_y):
#     for j in range(min_x, max_x):
#         matrix = image.Y[i, j]
#         matrix[0, 1:] = np.negative(matrix[0, 1:])
#         matrix[1:, :] = np.negative(matrix[1:, :])
#         image.Y[i, j] = matrix
#
# min_x_cb_cr = min_x // 2
# max_x_cb_cr = max_x // 2
# min_y_cb_cr = min_y // 2
# max_y_cb_cr = max_y // 2
#
# # Apply the operation to image.Cb
# for i in range(min_y_cb_cr, max_y_cb_cr):
#     for j in range(min_x_cb_cr, max_x_cb_cr):
#         matrix = image.Cb[i, j]
#         matrix[0, 1:] = np.negative(matrix[0, 1:])
#         matrix[1:, :] = np.negative(matrix[1:, :])
#         image.Cb[i, j] = matrix
#
# # Apply the operation to image.Cr
# for i in range(min_y_cb_cr, max_y_cb_cr):
#     for j in range(min_x_cb_cr, max_x_cb_cr):
#         matrix = image.Cr[i, j]
#         matrix[0, 1:] = np.negative(matrix[0, 1:])
#         matrix[1:, :] = np.negative(matrix[1:, :])
#         image.Cr[i, j] = matrix

# Generate a random matrix of 1 and -1
def generate_random_matrix(shape):
    return np.random.choice([1, -1], size=shape)


#   livello high, modifica DC e AC per tutte le componenti YUV
# for i in range(min_y, max_y):
#     for j in range(min_x, max_x):
#         matrix = image.Y[i, j]
#         random_matrix = generate_random_matrix(matrix.shape)
#         matrix = np.multiply(matrix, random_matrix)
#         image.Y[i, j] = matrix
#
# min_x_cb_cr = min_x // 2
# max_x_cb_cr = max_x // 2
# min_y_cb_cr = min_y // 2
# max_y_cb_cr = max_y // 2
#
# # Apply the operation to image.Cb
# for i in range(min_y_cb_cr, max_y_cb_cr):
#     for j in range(min_x_cb_cr, max_x_cb_cr):
#         matrix = image.Cb[i, j]
#         random_matrix = generate_random_matrix(matrix.shape)
#         matrix = np.multiply(matrix, random_matrix)
#         image.Cb[i, j] = matrix
#
# # Apply the operation to image.Cr
# for i in range(min_y_cb_cr, max_y_cb_cr):
#     for j in range(min_x_cb_cr, max_x_cb_cr):
#         matrix = image.Cr[i, j]
#         random_matrix = generate_random_matrix(matrix.shape)
#         matrix = np.multiply(matrix, random_matrix)
#         image.Cr[i, j] = matrix


# print(image.Y[min_y:max_y, min_x:max_x][0])


image.write_dct('output_scrambled.jpg')

# image_array = cv2.imread('output_scrambled.jpg')
#
# cv2.imwrite('output_scrambled.jpg', image_array, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
