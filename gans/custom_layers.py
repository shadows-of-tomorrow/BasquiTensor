import tensorflow as tf

from tensorflow.keras import backend
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Layer


class AdaptiveInstanceModulation(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        x_norm = self._instance_normalization(inputs[0])
        ys, yb = self._reshape_dense(inputs[1])
        return x_norm + (ys * x_norm + yb)

    @staticmethod
    def _reshape_dense(y):
        y = tf.reshape(y, [-1, 2, y.shape[1] // 2])
        return y[:, 0], y[:, 1]

    @staticmethod
    def _instance_normalization(x):
        x -= backend.mean(x, axis=[1, 2], keepdims=True)
        x *= (1.0 / backend.sqrt(backend.mean(backend.square(x), axis=[1, 2], keepdims=True) + 1e-8))
        return x


class NoiseModulation(Layer):

    def __init__(self, activation=None, **kwargs):
        self.activation = activation
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_variable(shape=[input_shape[3]], initializer='zeros', trainable=True)
        self.bias = self.add_variable(shape=[input_shape[3]], initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        noise = tf.random.normal(shape=[tf.shape(inputs)[0], inputs.shape[1], inputs.shape[2], 1])
        kernel = tf.reshape(self.kernel, [1, 1, 1, -1])
        bias = tf.reshape(self.bias, [1, 1, 1, -1])
        output = inputs + (kernel * noise + bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


class Constant(Layer):

    def __init__(self, shape, **kwargs):
        self.shape = shape
        super(Constant, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_variable(shape=self.shape, initializer='ones', trainable=True)
        super(Constant, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.tile(self.kernel, [tf.shape(inputs)[0], 1, 1, 1])


class PixelNormalization(Layer):

    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    def call(self, pix, **kwargs):
        pix_norm_const = backend.sqrt(backend.mean(backend.square(pix), axis=-1, keepdims=True) + 1e-08)
        return pix / pix_norm_const

    def compute_output_shape(self, input_shape):
        return input_shape


class MinibatchStDev(Layer):

    def __init__(self, **kwargs):
        super(MinibatchStDev, self).__init__(**kwargs)

    def call(self, pix, **kwargs):
        pix_mean = backend.mean(pix, axis=0, keepdims=True)
        pix_sqr_diff = backend.square(pix - pix_mean)
        pix_var = backend.mean(pix_sqr_diff, axis=0, keepdims=True) + 1e-08
        pix_stdev = backend.sqrt(pix_var)
        pix_mean_stdev = backend.mean(pix_stdev, keepdims=True)
        shape = backend.shape(pix)
        fmap_extra = backend.tile(pix_mean_stdev, (shape[0], shape[1], shape[2], 1))
        pix_extra = backend.concatenate([pix, fmap_extra], axis=-1)
        return pix_extra

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)


class WeightedSum(Add):

    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')

    def get_config(self):
        config = super().get_config().copy()
        config.update({'alpha': self.alpha.numpy()})
        return config

    def _merge_function(self, pixs):
        assert len(pixs) == 2
        return (1.0 - self.alpha) * pixs[0] + self.alpha * pixs[1]
