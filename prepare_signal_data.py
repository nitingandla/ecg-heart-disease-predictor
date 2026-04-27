import os
import cv2
import numpy as np

DATA_DIR = "data/train"

X, y = [], []

class_names = sorted(os.listdir(DATA_DIR))

for label, class_name in enumerate(class_names):
    folder = os.path.join(DATA_DIR, class_name)

    for img_name in os.listdir(folder):
        path = os.path.join(folder, img_name)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (32, 8))
        img = img / 255.0

        X.append(img)
        y.append(label)

X = np.array(X)  # (N, 32, 8)
y = np.array(y)

print("X shape:", X.shape)

np.save("X_multi.npy", X)
np.save("y_multi.npy", y)