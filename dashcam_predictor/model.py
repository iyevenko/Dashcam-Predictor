import os

import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, MaxPool2D
import dashcam_predictor.dataset as ds
import matplotlib.pyplot as plt

class PredictorModel(tf.keras.Model):

    def __init__(self, video_resolution=(None, None), segmented=False, classes=None, *args, **kwargs):
        super(PredictorModel, self).__init__(*args, **kwargs)

        output_filters = classes if segmented else 3
        w, h,  = video_resolution
        self.conv_LSTM = ConvLSTM2D(filters=64, kernel_size=(5, 5), input_shape=(None, h, w, 3), padding='same', return_sequences=False)
        self.batch_norm = BatchNormalization()
        self.conv2d_1 = Conv2D(filters=16, kernel_size=(9, 9), activation='relu', padding='same', data_format='channels_last')
        self.conv2d_2 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same', data_format='channels_last')
        self.conv2d_3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')
        self.conv2d_4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')
        self.conv2d_5 = Conv2D(filters=output_filters, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last')
        self.loss = tf.keras.losses.BinaryCrossentropy if segmented else tf.keras.losses.MSE

    def call(self, inputs, **kwargs):
        X = self.conv_LSTM(inputs)
        X = self.batch_norm(X)
        X = self.conv2d_1(X)
        X = self.conv2d_2(X)
        X = self.conv2d_3(X)
        X = self.conv2d_4(X)
        X = self.conv2d_5(X)
        return X

    def compile(self, optimizer='adadelta', loss=None, metrics=tf.keras.metrics.MSE, loss_weights=None, weighted_metrics=None,
                run_eagerly=None, **kwargs):
        if loss == None:
            loss = self.loss
        super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, **kwargs)


CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', "saved_models", "model1.ckpt")
SAVE_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', "saved_models", "cLSTM-MSE")

# def cross_entropy_loss(y_true, y_pred):
#     return -tf.math.reduce_sum(tf.matmul(y_true, tf.math.log(y_pred)) +
#                                tf.matmul(1-y_true, tf.math.log(1-y_pred)))