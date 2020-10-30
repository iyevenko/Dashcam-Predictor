from collections import deque

import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import time
import numpy as np



def predictor_dataset_fn(batch_size=16, lookback_amount=10, resolution=(160, 90), num_files=1):
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'bdd100k', 'videos', 'train')

    w, h = resolution

    files = os.listdir(DATA_PATH)
    num_files = min(num_files, len(files))
    files = files[:num_files]

    def process_frame(frame):
        frame_tensor = tf.convert_to_tensor(frame)
        frame_tensor = tf.image.rot90(frame_tensor)
        frame_tensor = tf.image.resize(frame_tensor, (h, w))
        frame_tensor = frame_tensor / 255.0
        return frame_tensor

    def init_buffer(capture):
        frame_buffer = deque(lookback_amount)
        for i in range(lookback_amount):
            _, frame = capture.read()
            frame_buffer.append(process_frame(frame))

        return frame_buffer

    def generator():
        video_idx = 0
        capture = cv2.VideoCapture(files[video_idx])
        grabbed = False
        frame_buffer = None

        while True:
            if not grabbed:
                if video_idx >= num_files:
                    break
                capture = cv2.VideoCapture(files[video_idx])
                frame_buffer = init_buffer(capture)
                video_idx += 1

            grabbed, frame = capture.read()

            X = tf.stack(frame_buffer)
            Y = process_frame(frame)

            yield (X, Y)

            frame_buffer.popleft()
            frame_buffer.append(output)

    dataset = tf.data.Dataset.from_generator(generator=generator,
                                             output_types=(tf.float16, tf.float16),
                                             output_shapes=([lookback_amount, h, w, 3], [h, w, 3]))
    # dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset_size = num_files * (1210 - lookback_amount)
    train_size = int(0.8 * int(dataset_size / batch_size))
    train_set = dataset.take(train_size)
    test_set = dataset.skip(train_size)

    partitions = {
        'train': train_set,
        'test': test_set
    }
    return partitions

if __name__ == '__main__':
    dataset = predictor_dataset_fn(10, 16, (256, 144), 1)
    iter = dataset['train'].as_numpy_iterator()
    # #0104a3f0-9294818d.mov
    (inputs, output) = next(iter)
    for i in range(10):
        img = (inputs[0][i]*255).astype(np.uint8)
        cv2.imshow('window', img)
        cv2.waitKey(0)
    img = (output[0] * 255).astype(np.uint8)
    cv2.imshow('window', img)
    cv2.waitKey(0)

    pass
