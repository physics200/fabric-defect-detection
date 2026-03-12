import os
import cv2
import numpy as np

image_folder = "dataset/images"
mask_folder = "dataset/masks"

for file in os.listdir(image_folder):

    mask_path = os.path.join(mask_folder, file)

    if not os.path.exists(mask_path):

        image_path = os.path.join(image_folder, file)
        img = cv2.imread(image_path)

        h, w, _ = img.shape

        empty_mask = np.zeros((h, w), dtype=np.uint8)

        cv2.imwrite(mask_path, empty_mask)

        print("Created mask for:", file)

print("All missing masks created.")