import tensorflow as tf

from tensorflow.keras import backend
from tensorflow.keras.layers import Layer


class AdaptiveInstanceModulation(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        x = self._instance_normalization(inputs[0])
        ys, yb = self._reshape_dense(inputs[1])
        return x + (ys * x + yb)

    def _reshape_dense(self, y):
        y = tf.reshape(y, [-1, 2, y.shape[1]//2])
        return y[:, 0], y[:, 1]

    def _instance_normalization(self, x):
        x -= backend.mean(x, axis=[1, 2], keepdims=True)
        x *= (1.0 / backend.sqrt(backend.mean(backend.square(x), axis=[1, 2], keepdims=True)+1e-8))
        return x


class NoiseModulation(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_variable(shape=[input_shape[3]], initializer='zeros', trainable=True)
        self.bias = self.add_variable(shape=[input_shape[3]], initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        noise = tf.random.normal(shape=[tf.shape(inputs)[0], inputs.shape[1], inputs.shape[2], 1])
        kernel = tf.reshape(self.kernel, [1, 1, 1, -1])
        bias = tf.reshape(self.bias, [1, 1, 1, -1])
        return inputs + (kernel * noise + bias)


class Constant(Layer):

    def __init__(self, shape, **kwargs):
        self.shape = shape
        super(Constant, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_variable(shape=self.shape, initializer='ones', trainable=True)
        super(Constant, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.tile(self.kernel, [tf.shape(inputs)[0], 1, 1, 1])
