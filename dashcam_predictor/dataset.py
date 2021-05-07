import os
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

DATA_PATH = os.path.join('data', 'bdd100k', 'videos')

def windowed_mov_ds(video_path, window_size, num_frames=100, skip_frames=1):
    cap = cv2.VideoCapture(video_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(num_frames, frameCount)
    # print(f'Frame count: {frameCount}')
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print('Allocating video memory')
    buf = np.empty((num_frames, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True
    print('Reading frames')
    while (fc < num_frames and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    @tf.function
    def process_frame(frame):
        frame = tf.image.rot90(frame)
        frame = tf.reverse(frame, axis=[-1])
        frame = tf.image.resize(frame, size=[224, 224])
        frame = frame / 255
        return frame

    print('Creating dataset')
    video_tensor = tf.map_fn(process_frame, buf, fn_output_signature=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices(video_tensor)

    ds = ds.window(size=window_size, shift=skip_frames, stride=skip_frames, drop_remainder=True)
    ds = ds.flat_map(lambda x: x.batch(window_size))
    prev = ds.map(lambda x: x[:-1,...])
    next = ds.map(lambda x: x[-1,...])
    ds = tf.data.Dataset.zip((prev, next))
    return ds

def dataset_from_mov(batch_size, window_size, train_file_num=None, val_file_num=None, test_file_num=None):
    train_files = os.listdir(os.path.join(DATA_PATH, 'train'))[:train_file_num]
    val_files = os.listdir(os.path.join(DATA_PATH, 'val'))[:val_file_num]
    test_files = os.listdir(os.path.join(DATA_PATH, 'test'))[:test_file_num]

    train_ds = None
    val_ds = None
    test_ds = None

    for video in train_files:
        ds = windowed_mov_ds(os.path.join(DATA_PATH, 'train', 'raw', video), window_size)
        if train_ds is None:
            train_ds = ds
        else:
            train_ds = train_ds.concatenate(ds)
        del ds

    for video in val_files:
        ds = windowed_mov_ds(os.path.join(DATA_PATH, 'val', 'raw', video), window_size)
        if val_ds is None:
            val_ds = ds
        else:
            val_ds = val_ds.concatenate(ds)
        del ds

    for video in test_files:
        ds = windowed_mov_ds(os.path.join(DATA_PATH, 'test', 'raw', video), window_size)
        if test_ds is None:
            test_ds = ds
        else:
            test_ds = test_ds.concatenate(ds)
        del ds

    if train_ds is not None:
        train_ds = train_ds.shuffle(10000).batch(batch_size, drop_remainder=True)
    if val_ds is not None:
        val_ds = val_ds.shuffle(10000).batch(batch_size, drop_remainder=True)
    if test_ds is not None:
        test_ds = test_ds.batch(batch_size, drop_remainder=True)

    dataset = {
        'train': train_ds,
        'val': val_ds,
        'test': test_ds
    }

    return dataset

def plot_frames(frames):
    window_size = frames.shape[0]
    fig = plt.figure(figsize=(2*window_size, 2))
    for i in range(window_size):
        fig.add_subplot(1, window_size, i+1)
        plt.tick_params(axis='x', bottom=False, labelbottom=False)
        plt.tick_params(axis='y', left=False, labelleft=False)
        plt.imshow(frames[i])

    plt.show()

def extract_jpgs(data_path):
    mov_path = os.path.join(data_path, 'raw')
    jpg_path = os.path.join(data_path, 'frames')
    mov_files = os.listdir(mov_path)

    for mov in mov_files:
        cap = cv2.VideoCapture(os.path.join(mov_path, mov))
        frame_dir = os.path.join(jpg_path, mov.split(os.extsep)[0])
        if not os.path.exists(frame_dir):
            os.mkdir(frame_dir)

        count = 0
        success, frame = cap.read()
        while success:
            frame_path = os.path.join(frame_dir, f'frame_{count:04}.jpg')
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(frame_path, frame)

            success, frame = cap.read()
            count += 1

        cap.release()

def dataset_from_jpgs(batch_size, window_size):
    train_path = os.path.join(DATA_PATH, 'train', 'frames')
    train_videos = [os.path.join(train_path, vid, '*.jpg') for vid in os.listdir(train_path)]
    train_ds = tf.data.Dataset.from_tensor_slices(train_videos)

    val_path = os.path.join(DATA_PATH, 'val', 'frames')
    val_videos = [os.path.join(val_path, vid, '*.jpg') for vid in os.listdir(val_path)]
    val_ds = tf.data.Dataset.from_tensor_slices(val_videos)

    test_path = os.path.join(DATA_PATH, 'test', 'frames')
    test_videos = [os.path.join(test_path, vid, '*.jpg') for vid in os.listdir(test_path)]
    test_ds = tf.data.Dataset.from_tensor_slices(test_videos)

    @tf.function
    def filename_to_frame_tensor(frame_file):
        frame_str = tf.io.read_file(frame_file)
        decoded_frame = tf.image.decode_jpeg(frame_str, channels=3)
        resized_image = tf.image.resize(decoded_frame, [224, 224])
        rescaled_image = resized_image / 255
        return rescaled_image

    @tf.function
    def make_windows(frames_dir):
        files_ds = tf.data.Dataset.list_files(frames_dir, shuffle=False)
        files_ds = files_ds.map(filename_to_frame_tensor)
        window_ds = files_ds.window(window_size, 2, 2)
        window_ds = window_ds.flat_map(lambda x: x.batch(window_size))
        prev = window_ds.map(lambda x: x[:-1,...])
        next = window_ds.map(lambda x: x[-1,...])
        window_ds = tf.data.Dataset.zip((prev, next))
        return window_ds

    train_ds = train_ds.flat_map(make_windows).shuffle(100).batch(batch_size, drop_remainder=True)
    val_ds = val_ds.flat_map(make_windows).shuffle(100).batch(batch_size, drop_remainder=True)
    test_ds = test_ds.flat_map(make_windows).shuffle(100).batch(batch_size, drop_remainder=True)

    dataset = {
        'train': train_ds,
        'val': val_ds,
        'test': test_ds
    }

    return dataset

if __name__ == '__main__':
    pass
    # extract_jpgs(os.path.join('..', DATA_PATH, 'train'))
    # extract_jpgs(os.path.join('..', DATA_PATH, 'val'))
    # extract_jpgs(os.path.join('..', DATA_PATH, 'test'))

    # ds_splits = dataset_from_jpgs(1, 8)
    #
    # ds_train = ds_splits['train']
    # ds_val = ds_splits['val']
    # ds_test = ds_splits['test']
    #
    # ds_list = ds_train.as_numpy_iterator()
    # while input() != 'q':
    #     x, y = next(ds_list)
    #     print((x.shape, y.shape))
    #     concat = tf.concat([x[0], y], 0)
    #     plot_frames(concat)


