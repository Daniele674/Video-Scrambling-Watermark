import jpeglib
import numpy as np
from blind_watermark import WaterMark

bwm1 = WaterMark(password_img=1, password_wm=1)
extracted_data = bwm1.extract('output_scrambled.jpg', wm_shape=118, mode='str')
print(extracted_data)

# Split the extracted_data string into a list of substrings
data_list = extracted_data.split()

data_list = [int(i) for i in data_list]

# Assign the first string to seed_number
seed_number = data_list[0]

# Assign the last four strings to face_region
face_region = data_list[-4:]

np.random.seed(seed_number)

min_x, max_x, min_y, max_y = face_region

# Read DCT coefficients from the scrambled JPEG file
scrambled_image = jpeglib.read_dct('output_scrambled.jpg')

num_to_flip = 60

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

# for i in range(min_y, max_y):
#     for j in range(min_x, max_x):
#         matrix = scrambled_image.Y[i, j]
#
#         # Get the shape of the matrix
#         shape = matrix.shape
#
#         # Generate random indices, excluding the DC coefficient at index 0,0
#         indices = np.random.choice(np.prod(shape) - 1, num_to_flip) + 1
#
#         # Convert the indices to 2D coordinates
#         coords = np.unravel_index(indices, shape)
#
#         # Flip the sign of the selected coefficients back
#         matrix[coords] *= -1
#
#         scrambled_image.Y[i, j] = matrix

# # Define the ranges for Y, Cb, and Cr
# min_x_cb_cr = min_x // 2
# max_x_cb_cr = max_x // 2
# min_y_cb_cr = min_y // 2
# max_y_cb_cr = max_y // 2
#
# # Apply the inverse operation to image.Y
# for i in range(min_y, max_y):
#     for j in range(min_x, max_x):
#         matrix = scrambled_image.Y[i, j]
#         matrix[0, 1:] = np.negative(matrix[0, 1:])
#         matrix[1:, :] = np.negative(matrix[1:, :])
#         scrambled_image.Y[i, j] = matrix
#
# # Apply the inverse operation to image.Cb
# for i in range(min_y_cb_cr, max_y_cb_cr):
#     for j in range(min_x_cb_cr, max_x_cb_cr):
#         matrix = scrambled_image.Cb[i, j]
#         matrix[0, 1:] = np.negative(matrix[0, 1:])
#         matrix[1:, :] = np.negative(matrix[1:, :])
#         scrambled_image.Cb[i, j] = matrix
#
# # Apply the inverse operation to image.Cr
# for i in range(min_y_cb_cr, max_y_cb_cr):
#     for j in range(min_x_cb_cr, max_x_cb_cr):
#         matrix = scrambled_image.Cr[i, j]
#         matrix[0, 1:] = np.negative(matrix[0, 1:])
#         matrix[1:, :] = np.negative(matrix[1:, :])
#         scrambled_image.Cr[i, j] = matrix

# Write descrambled coefficients back to a JPEG file
scrambled_image.write_dct('output_descrambled.jpg')
