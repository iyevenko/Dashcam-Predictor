import os
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, Conv2DTranspose, BatchNormalization, LSTM, Dense, Flatten, Reshape, Activation

# CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', "saved_models", "model1.ckpt")
SAVE_MODEL_PATH = os.path.join("saved_models", "cLSTM-MSE-Skip")
# SAVE_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', "saved_models", "VPGAN-ResNet50")


class DeterministicModel(tf.keras.Model):
    def __init__(self, num_layers, num_filters=32, *args, **kwargs):
        super(DeterministicModel, self).__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.num_filters = num_filters

        self.conv_lstm_layers = [
            ConvLSTM2D(filters=num_filters, kernel_size=(5, 5), input_shape=(None, 224, 224, 3 if i == 0 else num_filters),
                       padding='same', return_sequences=True)
            for i in range(num_layers)
        ]

        self.final_conv_lstm = ConvLSTM2D(filters=num_filters, kernel_size=(5, 5), input_shape=(None, 224, 224, num_filters),
                                          padding='same', return_sequences=False)
        self.final_conv = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='sigmoid')

        self.batchnorms = [
            BatchNormalization()
            for _ in range(num_layers+1)
        ]

    def call(self, inputs, **kwargs):
        X = inputs
        paddings = np.zeros((5, 2), dtype=np.int32)
        paddings[-1, 1] = self.num_filters - 3
        prevX = tf.pad(X, paddings)

        for i in range(self.num_layers):
            X = self.conv_lstm_layers[i](X)
            X += prevX
            prevX = X

        X = self.final_conv_lstm(X)
        paddings = np.zeros((4, 2), dtype=np.int32)
        paddings[-1, 1] = self.num_filters - 3
        prevX = tf.pad(inputs[:,-1,...], paddings)
        X += prevX
        X = self.final_conv(X)

        return X

    def compile(self, optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.MSE, metrics=tf.keras.metrics.MSE, loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None, **kwargs):
        super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, **kwargs)


""" 
VPGAN Implementation with ResNet50 for transfer learning
"""


class NormalizedDense(tf.keras.layers.Layer):

    def __init__(self, units, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.dense = tf.keras.layers.Dense(units, use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU()

    def call(self, inputs, **kwargs):
        X = self.dense(inputs)
        X = self.batchnorm(X)
        X = self.leaky_relu(X)
        return X


class PooledConv2D(tf.keras.layers.Layer):
    NOISE_STDDEV = 0.5
    def __init__(self, filters, kernel_size, noisy=False, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.noisy = noisy
        self.conv = Conv2D(filters, kernel_size, padding='same', data_format='channels_last', use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.gaussian_noise = tf.keras.layers.GaussianNoise(self.NOISE_STDDEV)
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.pooling = tf.keras.layers.AveragePooling2D()

    def call(self, inputs, **kwargs):
        X = self.conv(inputs)
        X = self.batchnorm(X)
        if self.noisy:
            X = self.gaussian_noise(X)
        X = self.leaky_relu(X)
        X = self.pooling(X)
        return X


class NormalizedConvTranspose2D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU()

    def call(self, inputs, **kwargs):
        X = self.conv_transpose(inputs)
        X = self.batchnorm(X)
        X = self.leaky_relu(X)
        return X


# Encodes the previous frames and variances into a tensor
# with mean and log variance concatenated for sampling Z_t
class FrameEncoder(tf.keras.Model):

    def __init__(self, buf_size, h, w, latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_encoder = tf.keras.Sequential([
            # (b, n, h, w, 3) -> (b, x)
            ConvLSTM2D(filters=32, kernel_size=(5, 5), padding='same', input_shape=(buf_size, h, w, 3), return_sequences=False),
            BatchNormalization(),
            PooledConv2D(16, (9, 9)),
            PooledConv2D(32, (5, 5)),
            PooledConv2D(32, (5, 5)),
            PooledConv2D(64, (3, 3)),
            PooledConv2D(64, (3, 3)),
            Flatten(),
            NormalizedDense(512),
            NormalizedDense(512),
            Dense(512)
        ])
        self.z_encoder = tf.keras.Sequential([
            # (b, n, d) -> (b, y)
            LSTM(256, return_sequences=False),
            BatchNormalization(),
            NormalizedDense(512),
            NormalizedDense(512),
            Dense(512)
        ])
        self.mean_generator = tf.keras.Sequential([
            # (b, x+y) -> (b, z)
            NormalizedDense(512),
            NormalizedDense(512),
            NormalizedDense(512),
            NormalizedDense(latent_dim)
        ])
        self.sd_generator = tf.keras.Sequential([
            # (b, x+y) -> (b, z)
            NormalizedDense(512),
            NormalizedDense(latent_dim)
        ])

    def call(self, inputs, training=None, mask=None):
        X, Z = inputs
        X_enc = self.x_encoder(X)
        Z_enc = self.z_encoder(Z)

        concat = tf.keras.layers.Concatenate()([X_enc, Z_enc])

        mu = self.mean_generator(concat)
        sigma = self.sd_generator(concat)

        output = tf.keras.layers.Concatenate(axis=1)([mu, sigma])

        return output


# Decodes the previous frame and variance into a tensor
# with mean and log variance concatenated for sampling X_t
class FrameDecoder(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_encoder = tf.keras.Sequential([
            # (b, 224, 224, 3) -> (b, x)
            PooledConv2D(16, (9, 9)),
            PooledConv2D(32, (5, 5)),
            PooledConv2D(32, (5, 5)),
            PooledConv2D(64, (3, 3)),
            PooledConv2D(64, (3, 3)),
            Flatten(),
            NormalizedDense(512),
            Dense(256)
        ])
        self.z_encoder = tf.keras.Sequential([
            # (b, d) -> (b, y)
            NormalizedDense(512),
            NormalizedDense(512),
            NormalizedDense(512),
            NormalizedDense(512),
            Dense(256)
        ])
        self.mean_generator = tf.keras.Sequential([
            # (b, x+y) -> (b, 224, 224, 3)
            NormalizedDense(512),
            NormalizedDense(7 * 7 * 64),
            Reshape((7, 7, 64)),
            NormalizedConvTranspose2D(128, (5, 5), (1, 1)),
            NormalizedConvTranspose2D(64, (5, 5), (4, 4)),
            NormalizedConvTranspose2D(32, (5, 5), (4, 4)),
            Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            BatchNormalization(),
            Activation('tanh')
        ])
        self.sd_generator = tf.keras.Sequential([
            # (b, x+y) -> (b, 224, 224, 3)
            NormalizedDense(512),
            NormalizedDense(7 * 7 * 16),
            Reshape((7, 7, 16)),
            NormalizedConvTranspose2D(32, (5, 5), (1, 1)),
            NormalizedConvTranspose2D(16, (5, 5), (4, 4)),
            NormalizedConvTranspose2D(16, (5, 5), (4, 4)),
            Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            BatchNormalization(),
            Activation('tanh')
        ])

    def call(self, inputs, training=None, mask=None):
        Z_fake, X_prev = inputs
        X_enc = self.x_encoder(X_prev)
        Z_enc = self.z_encoder(Z_fake)

        concat = tf.keras.layers.Concatenate()([X_enc, Z_enc])

        mu = self.mean_generator(concat)
        sigma = self.sd_generator(concat)

        output = tf.keras.layers.Concatenate(axis=1)([mu, sigma])

        return output


class Discriminator(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_encoder = tf.keras.Sequential([
            # (b, 224, 224, 3) -> (b, x)
            PooledConv2D(16, (9, 9), noisy=True),
            PooledConv2D(32, (5, 5), noisy=True),
            PooledConv2D(32, (5, 5), noisy=True),
            PooledConv2D(64, (3, 3), noisy=True),
            PooledConv2D(64, (3, 3), noisy=True),
            Flatten(),
            NormalizedDense(512),
            Dense(512)
        ])
        self.z_encoder = tf.keras.Sequential([
            # (b, d) -> (b, y)
            NormalizedDense(512),
            NormalizedDense(512),
            NormalizedDense(512),
            NormalizedDense(512),
            Dense(512)
        ])
        self.discriminator = tf.keras.Sequential([
            # (b, x+y) -> (b, 1)
            NormalizedDense(1024),
            NormalizedDense(512),
            NormalizedDense(64),
            Dense(1, activation='sigmoid')
        ])

    def call(self, inputs, training=None, mask=None):
        X, Z = inputs
        X = self.x_encoder(X)
        Z = self.z_encoder(Z)

        concat = tf.keras.layers.Concatenate()([X, Z])
        output = self.discriminator(concat)

        return output


class GenerativeModel(tf.keras.Model):
    W, H = (224, 224)
    def __init__(self, latent_dim, buf_size, alpha=1, beta=1, gamma=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.latent_dim = latent_dim
        self.buf_size = buf_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.encoder = FrameEncoder(self.buf_size, self.H, self.W, self.latent_dim)
        self.decoder = FrameDecoder()
        self.discriminator = Discriminator()

        self.z_buffer = deque()
        self.z_buffer.append(tf.zeros((self.latent_dim)))
        # For extracting the kth layer representation from resnet
        self.resnet50_encoder = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', pooling='avg')
        self.resnet50_encoder.trainable = False

    def encode(self, X, Z):
        mean, log_var = tf.split(self.encoder((X, Z)), num_or_size_splits=2, axis=1)
        xi = tf.random.normal(shape=mean.shape)
        Z_t = mean + log_var * xi
        return Z_t

    def decode(self, Z_fake, X_prev):
        mean, log_var = tf.split(self.decoder((Z_fake, X_prev)), num_or_size_splits=2, axis=1)
        xi = tf.random.normal(shape=mean.shape)
        X_t = mean + log_var * xi
        return X_t

    def evaluate_fake(self, X, Z):
        return self.discriminator((X, Z))

    def cross_entropy(self, y_true, y_pred):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)

    # Z_fake = tf.random.normal(shape=self.latent_dim)
    def adv_loss(self, X_t, Z_t, X_fake, Z_fake):
        # E(logD(X_t, G_psi(X, Z))) + E(1 - logD(G_theta(Z'_t, X_t-1), Z'_t))
        real = self.evaluate_fake(X_t, Z_t)
        fake = self.evaluate_fake(X_fake, Z_fake)
        L_adv = self.cross_entropy(tf.ones_like(real), real) + self.cross_entropy(tf.zeros_like(fake), fake)

        return L_adv

    # Only considers one-step cycle consistency
    def cycle_loss(self, X_prev, X_t, Z_t):
        # E(|X_t - G_theta(Z_t, G_theta(-Z_t, X_t))|) + E(|X_t-1 - G_theta(-Z_t, G_theta(Z_t, X_t))|)
        L_cycle = tf.reduce_mean(tf.abs(X_t - self.decode(Z_t, self.decode(-Z_t, X_t))) +
                                 tf.abs(X_prev - self.decode(-Z_t, self.decode(Z_t, X_t))))
        return L_cycle

    def perceptual_loss(self, X_t, X_gen):
        assert X_t.shape[-3:] == (self.H, self.W, 3)
        assert X_gen.shape[-3:] == (self.H, self.W, 3)
        return tf.keras.losses.MSE(self.resnet50_encoder(X_t), self.resnet50_encoder(X_gen))

    # Just using one-step recon loss, can be changed to k-step in the future
    def recon_loss(self, X_prev, X_t, Z_t):
        # E(phi(X_t, G_theta(Z_t, X_t-1)))
        X_gen = self.decode(Z_t, X_prev)
        L1 = tf.reduce_sum(tf.abs(X_t - X_gen))
        Lp = self.perceptual_loss(X_t, X_gen)
        return tf.reduce_mean(L1 + Lp)

    # Recursively calculate Z_(t-n:t) and stores in Z tensor
    def calculate_Z(self, X):
        batch_size = X.shape[0]
        buf_size = X.shape[1]

        # Note sure what to initialize Z_0 with
        Z = tf.random.normal((batch_size, 1, self.latent_dim))

        for i in range(buf_size):
            # Z_next -> (b, 1, d),
            Z_next = tf.expand_dims(self.encode(X[:, :i + 1, :, :, :], Z), 1)
            # Z -> (b, i+2, d)
            Z = tf.concat([Z, Z_next], 1)
        # Z -> (b, buf_size+1, d)
        return Z

    # Somewhat abusing the parameters sent to the loss function by the Model Class
    # Instead of y_true and y_pred, passing x and y_true makes calculations easier
    def loss(self, X, X_t):
        # X_prev -> (b, h, w, 3)
        X_prev = X[:, -1, :, :, :]
        

        # Z -> (b, buf_size+1, d)
        Z = self.calculate_Z(X)
        # Z_t -> (b, n)
        Z_t = Z[:, -1, :]
        # Z_fake -> (b, n)
        Z_fake = tf.random.normal(shape=Z_t.shape)
        # X_fake -> (b, h, w, 3)
        X_fake = self.decode(Z_fake, X_prev)

        L_adv = self.adv_loss(X_t, Z_t, X_fake, Z_fake)
        L_cycle = self.cycle_loss(X_prev, X_t, Z_t)
        L_recon = self.recon_loss(X_prev, X_t, Z_t)

        self.add_metric(L_adv, name='Adversarial Loss')
        self.add_metric(L_cycle, name='Cycle Loss')
        self.add_metric(L_recon, name='Reconstruction Loss')

        return (self.alpha * L_adv) + (self.beta * L_cycle) + (self.gamma * L_recon)

    def call(self, inputs, training=None, mask=None):
        X = inputs
        Z = self.calculate_Z(X)
        print(Z.shape)
        print(tf.reduce_max(Z))
        X_prev = X[:, -1, :, :, :]
        Z_t = Z[:, -1, :]
        X_t = self.decode(Z_t, X_prev)
        print(X_t.shape)
        print(tf.reduce_max(X_t))
        return X_t

    def train_step(self, data):
        # X -> (b, n, h, w, 3), X_t -> (b, h, w, 3)
        X, X_t = data

        with tf.GradientTape() as tape:
            loss = self.compiled_loss(X, X_t)

        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.compiled_metrics.update_state(X, X_t)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # X -> (b, n, h, w, 3), X_t -> (b, h, w, 3)
        X, X_t = data

        self.compiled_loss(X, X_t)
        self.compiled_metrics.update_state(X, X_t)

        return {m.name: m.result() for m in self.metrics}

    def compile(self, optimizer='adam', loss=None, metrics=None, loss_weights=None, weighted_metrics=None,
                run_eagerly=None, **kwargs):
        super().compile(optimizer, self.loss, metrics, loss_weights, weighted_metrics, run_eagerly, **kwargs)
