import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 256

# load trained model
model = tf.keras.models.load_model("fabric_defect_model.keras")

# choose any image from dataset
image_path = "dataset/images/1182.PNG"

img = cv2.imread(image_path)
img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

input_img = img_resized / 255.0
input_img = np.expand_dims(input_img, axis=0)

# prediction
prediction = model.predict(input_img)[0]


# create binary mask
mask = prediction[:,:,0] > 0.5

# create overlay image
overlay = img_resized.copy()
overlay[mask] = [255, 0, 0]   # highlight defect in red


plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))

plt.subplot(1,3,2)
plt.title("Predicted Mask")
plt.imshow(prediction[:,:,0], cmap="gray")

plt.subplot(1,3,3)
plt.title("Detected Defect")
plt.imshow(prediction[:,:,0] > 0.5, cmap="gray")

plt.show()