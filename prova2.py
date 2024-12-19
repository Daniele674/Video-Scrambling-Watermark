from imwatermark import WatermarkEncoder, WatermarkDecoder
import cv2

img = cv2.imread('testSet/persona.jpg')

encoder = WatermarkEncoder()
encoder.loadModel()
encoder.set_watermark('bytes', 'ciao'.encode('utf-8'))
img_encoded = encoder.encode(img, 'dwtDct')
cv2.imwrite('frame.jpg', img_encoded, params=[cv2.IMWRITE_JPEG_QUALITY, 100])

img = cv2.imread('frame.jpg')
decoder = WatermarkDecoder('bytes', 32)
# Decode the watermark from the image
watermark = decoder.decode(img, 'dwtDct')

print(watermark.decode('utf-8'))