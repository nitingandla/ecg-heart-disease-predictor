import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   
    resized = cv2.resize(gray, (32, 8))

    normalized = resized / 255.0

    return np.array(normalized, dtype=np.float32)