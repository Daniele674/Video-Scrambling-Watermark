import cv2
from turbojpeg import TurboJPEG
import numpy as np


jpeg = TurboJPEG(r"C:\libjpeg-turbo64\bin\turbojpeg.dll")

in_file = open('testSet/persona.jpg', 'rb')
image_data = in_file.read()
in_file.close()

image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

encoded_image = jpeg.encode(image_array, quality=100)  # Example with quality set to 80

out_file = open('output.jpg', 'wb')
out_file.write(encoded_image)
out_file.close()
