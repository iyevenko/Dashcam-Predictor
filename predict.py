import os
from collections import deque

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dashcam_predictor.model as model
import dashcam_predictor.dataset as ds
import cv2

w, h = (160, 90)
LOOKBACK_AMT = 8

loaded_model = tf.keras.models.load_model(model.SAVE_MODEL_PATH)

frame_buffer = deque(maxlen=LOOKBACK_AMT)
video_path = os.path.join(ds.DATA_PATH, os.listdir(ds.DATA_PATH)[0])
capture = cv2.VideoCapture(video_path)

for i in range(LOOKBACK_AMT):
    _, frame = capture.read()
    frame_tensor = tf.convert_to_tensor(frame)
    frame_tensor = tf.image.rot90(frame_tensor)
    frame_tensor = tf.image.resize(frame_tensor, (h, w))
    frame_tensor = frame_tensor / 255.0
    frame_buffer.append(frame_tensor)

while True:
    grabbed, actual = capture.read()
    if (not grabbed):
        break

    buffer_tensor = tf.expand_dims(tf.stack(frame_buffer), axis=0)
    prediction = loaded_model.predict(buffer_tensor)
    adjusted_prediction = cv2.resize(prediction[0]*255, dsize=(2*w, 2*h)).astype(np.uint8)

    adjusted_actual = cv2.rotate(actual, rotateCode=2)
    adjusted_actual = cv2.resize(adjusted_actual, dsize=(2*w, 2*h))

    cv2.imshow('Current Frame', adjusted_prediction)
    cv2.imshow('Actual Frame', adjusted_actual)
    cv2.waitKey(0)

    frame_tensor = tf.convert_to_tensor(actual)
    frame_tensor = tf.image.rot90(frame_tensor)
    frame_tensor = tf.image.resize(frame_tensor, (h, w))
    frame_tensor = frame_tensor / 255.0
    frame_buffer.popleft()
    frame_buffer.append(frame_tensor)
