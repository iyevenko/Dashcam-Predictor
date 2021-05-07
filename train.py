import datetime
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from dashcam_predictor.dataset import *
from dashcam_predictor.model import *

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], False)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


if __name__ == '__main__':
    batch_size = 8
    buf_size = 4
    resolution = (224, 224) # Required for Resnet50
    # latent_dim = 256

    # dataset = dataset_from_mov(batch_size, buf_size, 5, 1, 0)
    dataset = dataset_from_jpgs(batch_size, buf_size)
    model = DeterministicModel(2)
    model.compile()

    # ds_iter = dataset['test'].__iter__()
    # x, y = next(ds_iter)
    # print(model(x).shape)

    log_dir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=10)
    model.fit(dataset['train'], epochs=1, steps_per_epoch=1000, callbacks=[tensorboard_callback],
              validation_data=dataset['val'], validation_steps = 100)

    tf.keras.models.save_model(model, SAVE_MODEL_PATH)