import os

import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D
import dataset as ds

def model_fn(resolution):
    h, w, = resolution
    model = tf.keras.models.Sequential([
        ConvLSTM2D(filters=40, kernel_size=(5, 5), input_shape=(None, h, w, 3), padding='same', return_sequences=True),
        BatchNormalization(),
        ConvLSTM2D(filters=40, kernel_size=(5, 5), padding='same', return_sequences=True),
        BatchNormalization(),
        ConvLSTM2D(filters=40, kernel_size=(5, 5), padding='same', return_sequences=True),
        BatchNormalization(),
        ConvLSTM2D(filters=40, kernel_size=(5, 5), padding='same', return_sequences=False),
        BatchNormalization(),
        Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last')
    ])
    return model

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', "saved_models", "model1.ckpt")

if __name__ == '__main__':
    resolution = (426, 240) #240p as opposed to 720p
    dataset = ds.input_fn(lookback_amount=10, batch_size=16, resolution=resolution, num_files=1)

    model = model_fn(resolution)
    model.compile(loss = tf.keras.losses.binary_crossentropy, optimizer='adadelta', metrics=['accuracy'])
    model.summary()

    callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1)
    model.fit(dataset['train'], epochs=10, callbacks=[callback])