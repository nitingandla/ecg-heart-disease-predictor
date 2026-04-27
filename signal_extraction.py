import cv2
import numpy as np

def image_to_multichannel_signal(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image not loaded properly")

    img = cv2.resize(img, (256, 256))

    img = img / 255.0

    signal = []
    for i in range(8):
        shifted = np.roll(img, shift=i*5, axis=1)
        signal.append(shifted.flatten())

    signal = np.array(signal).T 

    return img, signal