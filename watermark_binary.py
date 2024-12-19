# encoding
import cv2
from LSBSteg import LSBSteg
import chardet


steg = LSBSteg(cv2.imread("testSet/pulcino.png"))
data = open("watermark_informations.bin", "rb").read()
new_img = steg.encode_binary(data)
cv2.imwrite("new_image.png", new_img)

# decoding
steg = LSBSteg(cv2.imread("new_image.png"))
binary = steg.decode_binary()
with open("recovered.bin", "wb") as f:
    f.write(binary)

# Open the file in binary read mode
with open("recovered.bin", "rb") as file:
    content = file.read()

detected = chardet.detect(content)
encoding = detected['encoding']
text_content = content.decode(encoding)

# Find the start of the list
start_index = text_content.find('[')

# Slice the string from the start of the list
text_content = text_content[start_index:]

text_content = text_content.replace('î.', '')

# Now 'text_content' contains the list data
print(text_content)
