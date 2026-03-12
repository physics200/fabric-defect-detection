# import os
# import cv2
# import matplotlib.pyplot as plt

# image_folder = "dataset/images"
# mask_folder = "dataset/masks"

# # pick one file
# image_files = os.listdir(image_folder)
# file = image_files[0]

# image_path = os.path.join(image_folder, file)
# mask_path = os.path.join(mask_folder, file)

# image = cv2.imread(image_path)
# mask = cv2.imread(mask_path, 0)

# plt.figure(figsize=(10,5))

# plt.subplot(1,2,1)
# plt.title("Fabric Image")
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# plt.subplot(1,2,2)
# plt.title("Defect Mask")
# plt.imshow(mask, cmap="gray")

# plt.show()


# import os
# import cv2
# import numpy as np

# image_folder = "dataset/images"
# mask_folder = "dataset/masks"

# IMG_SIZE = 256

# images = []
# masks = []

# files = os.listdir(image_folder)

# print("Total files:", len(files))

# for file in files:

#     img_path = os.path.join(image_folder, file)
#     mask_path = os.path.join(mask_folder, file)

#     img = cv2.imread(img_path)
#     mask = cv2.imread(mask_path, 0)

#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

#     images.append(img)
#     masks.append(mask)

# images = np.array(images) / 255.0
# masks = np.array(masks) / 255.0

# masks = np.expand_dims(masks, axis=-1)

# print("Images shape:", images.shape)
# print("Masks shape:", masks.shape)





import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

IMG_SIZE = 256

image_folder = "dataset/images"
mask_folder = "dataset/masks"

images = []
masks = []

for file in os.listdir(image_folder):

    img = cv2.imread(os.path.join(image_folder, file))
    mask = cv2.imread(os.path.join(mask_folder, file), 0)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

    images.append(img)
    masks.append(mask)

images = np.array(images) / 255.0
masks = np.array(masks) / 255.0

masks = np.expand_dims(masks, axis=-1)


def build_unet():

    inputs = layers.Input((256,256,3))

    c1 = layers.Conv2D(32,3,activation='relu',padding='same')(inputs)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64,3,activation='relu',padding='same')(p1)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128,3,activation='relu',padding='same')(p2)

    u1 = layers.UpSampling2D()(c3)
    concat1 = layers.Concatenate()([u1,c2])
    c4 = layers.Conv2D(64,3,activation='relu',padding='same')(concat1)

    u2 = layers.UpSampling2D()(c4)
    concat2 = layers.Concatenate()([u2,c1])
    c5 = layers.Conv2D(32,3,activation='relu',padding='same')(concat2)

    outputs = layers.Conv2D(1,1,activation='sigmoid')(c5)

    model = models.Model(inputs,outputs)

    return model


model = build_unet()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(
    images,
    masks,
    batch_size=8,
    epochs=13
)
loss, accuracy = model.evaluate(images, masks)

print("Model Accuracy: {:.2f}%".format(accuracy * 100))
model.save("fabric_defect_model.keras")

