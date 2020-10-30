import matplotlib.pyplot as plt
import tensorflow as tf

from dashcam_predictor.dataset import PredictorDataset
from dashcam_predictor.model import PredictorModel

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

resolution = (160, 90) #144p  as opposed to 720p
dataset = PredictorDataset( batch_size=8, lookback_amount=15, resolution=resolution, num_files=1)

model = PredictorModel(resolution)
model.compile()
model.summary()

history = model.fit(dataset['train'], epochs=10)
loss_plot = plt.plot(history.history['loss'])
plt.show()

loss = model.evaluate(x=dataset['test'])

print('Test Loss: {}'.format(loss))

model.save(dataset.SAVE_MODEL_PATH)