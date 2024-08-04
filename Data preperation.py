import cv2
import numpy as np
from keras.utils import to_categorical
import os
from config import LABEL_DICT

data_path = 'E:\dataset'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]
label_dict = LABEL_DICT  # Use the label_dict from config

img_size = 100
data = []
target = []

for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)
    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (img_size, img_size))
        data.append(resized)
        target.append(labels[categories.index(category)])  # Map category to numeric label

data = np.array(data) / 255.0
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
target = np.array(target)
target = to_categorical(target, num_classes=len(label_dict))

np.save('data.npy', data)
np.save('target.npy', target)
