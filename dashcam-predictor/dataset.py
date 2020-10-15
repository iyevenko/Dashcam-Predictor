from collections import deque

import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras_video import VideoFrameGenerator
import time

DATA_PATH = '/Users/iyevenko/Documents/GitHub/Dashcam-Predictor/data/bdd100k/videos/train'

## Patch fix for dataset memeory leak
class TfDataset(object):
    def __init__(self):
        self.py_func_set_to_cleanup = set()

    def from_generator(self, generator, output_types, output_shapes=None, args=None):
        if not hasattr(tf.compat.v1.get_default_graph(), '_py_funcs_used_in_graph'):
            tf.compat.v1.get_default_graph()._py_funcs_used_in_graph = []
        py_func_set_before = set(tf.compat.v1.get_default_graph()._py_funcs_used_in_graph)
        result = tf.data.Dataset.from_generator(generator, output_types, output_shapes, args)
        py_func_set_after = set(tf.compat.v1.get_default_graph()._py_funcs_used_in_graph) - py_func_set_before
        self.py_func_set_to_cleanup |= py_func_set_after
        return result

    def cleanup(self):
        new_py_funcs = set(tf.compat.v1.get_default_graph()._py_funcs_used_in_graph) - self.py_func_set_to_cleanup
        tf.compat.v1.get_default_graph()._py_funcs_used_in_graph = list(new_py_funcs)
        self.py_func_set_to_cleanup = set()

def input_fn(lookback_amount, batch_size, resolution, num_files):
    h, w, = resolution
    num_files = min(len(os.listdir(DATA_PATH)), num_files)

    def generator():
        DATA_PATH = '/Users/iyevenko/Documents/GitHub/Dashcam-Predictor/data/bdd100k/videos/train'
        grabbed = False
        capture = None
        video_idx = 0
        video_files = os.listdir(DATA_PATH)
        frame_buffer = deque(maxlen=lookback_amount)

        while True:
            if not grabbed:
                if video_idx >= num_files or video_idx >= len(video_files):
                    break
                video = video_files[video_idx]
                video_path = os.path.join(DATA_PATH, video)
                capture = cv2.VideoCapture(video_path)
                video_idx += 1

                for i in range(lookback_amount):
                    _, frame = capture.read()
                    frame_tensor = tf.reverse(frame, axis=[-1])  # BGR -> RGB
                    frame_tensor = tf.image.rot90(frame_tensor)
                    frame_tensor = tf.image.resize(frame_tensor, resolution)
                    frame_buffer.append(frame_tensor)

            grabbed, frame = capture.read()

            if grabbed:
                frame_tensor = tf.reverse(frame, axis=[-1]) # BGR -> RGB
                frame_tensor = tf.image.rot90(frame_tensor)
                frame_tensor = tf.image.resize(frame_tensor, resolution)

                yield (tf.stack(frame_buffer), frame_tensor)

                # frame_buffer.popleft()
                frame_buffer.append(frame_tensor)

    # tf_dataset = TfDataset()
    # dataset = tf_dataset.from_generator(generator=generator,
    #                                     output_types=(tf.int32, tf.int32),
    #                                     output_shapes=([lookback_amount, h, w, 3], [h, w, 3]))

    dataset = tf.data.Dataset.from_generator(generator=generator,
                                             output_types=(tf.int32, tf.int32),
                                             output_shapes=([lookback_amount, h, w, 3], [h, w, 3]))
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset_size = num_files * (1210-lookback_amount)
    train_size = int(0.8 * int(dataset_size / batch_size))
    train_set = dataset.take(train_size)
    test_set = dataset.skip(train_size)

    del dataset
    # tf_dataset.cleanup()

    partitions = {
        'train': train_set,
        'test': test_set
    }
    return partitions


def video_generator_input_fn(lookback_frames, batch_size):
    glob_pattern = os.path.join(DATA_PATH, '*.mov')

    generator = VideoFrameGenerator(
        rescale=1/255.,
        nb_frames=lookback_frames+1,
        batch_size=batch_size,
        use_frame_cache=False,
        target_shape=(720, 1280),
        nb_channel=3,
        shuffle=False,
        glob_pattern=glob_pattern
    )

    dataset = tf.data.Dataset.from_generator(generator, output_types=tf.float32, output_shapes=[lookback_frames+1, 720, 1280, 3])

    return dataset


if __name__ == '__main__':
    # dataset = input_fn(10, 16, (426, 240), 1)
    # iter = dataset['train'].as_numpy_iterator()
    # # #0104a3f0-9294818d.mov
    # img = next(iter)
    # plt.imshow(img[1]-img[0][0])
    # plt.show()
    pass
