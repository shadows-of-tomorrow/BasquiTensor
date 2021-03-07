import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal


class DenseEQL(Layer):

    def __init__(self, units, gain=2, lrmul=1, **kwargs):
        super(DenseEQL, self).__init__(kwargs)
        self.units = units
        self.gain = gain
        self.lrmul = lrmul

    def get_config(self):
        config = super().get_config().copy()
        config.update({'units': self.units})
        config.update({'gain': self.gain})
        config.update({'lrmul': self.lrmul})
        return config

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.kernel = self.add_weight(
            shape=[self.in_channels, self.units],
            initializer=RandomNormal(mean=0.0, stddev=1.0 / self.lrmul),
            trainable=True,
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        fan_in = self.in_channels
        self.scale = tf.sqrt(self.gain/fan_in)

    def call(self, inputs):
        scale = tf.cast(self.scale, self.kernel.dtype)
        output = tf.matmul(inputs, scale * self.kernel) + self.bias
        return output * self.lrmul


class Conv2DEQL(Layer):

    def __init__(self, n_channels, kernel_size=3, gain=2, **kwargs):
        super(Conv2DEQL, self).__init__(kwargs)
        self.kernel_size = kernel_size
        self.out_channels = n_channels
        self.gain = gain

    def get_config(self):
        config = super().get_config().copy()
        config.update({'kernel_size': self.kernel_size})
        config.update({'n_channels': self.out_channels})
        config.update({'gain': self.gain})
        return config

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        initializer = RandomNormal(mean=0.0, stddev=1.0)
        self.kernel = self.add_weight(
            shape=[self.kernel_size, self.kernel_size, self.in_channels, self.out_channels],
            initializer=initializer,
            trainable=True,
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.out_channels,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        fan_in = self.kernel_size * self.kernel_size * self.in_channels
        self.scale = tf.sqrt(self.gain/fan_in)

    def call(self, inputs):
        scale = tf.cast(self.scale, self.kernel.dtype)
        output = tf.nn.conv2d(inputs, scale * self.kernel, strides=1, padding="SAME") + self.bias
        return output


class AdaptiveInstanceModulation(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        x_norm = self._instance_normalization(inputs[0])
        ys, yb = self._reshape_dense(inputs[1])
        output = x_norm + (ys * x_norm + yb)
        return output

    @staticmethod
    def _reshape_dense(y):
        y = tf.reshape(y, [-1, 2, 1, y.shape[1] // 2])
        ys = y[:, 0:1, :, :]
        yb = y[:, 1:2, :, :]
        return ys, yb

    @staticmethod
    def _instance_normalization(x):
        x_mean = backend.mean(x, axis=[1, 2], keepdims=True)
        x -= x_mean
        x_std = backend.sqrt(backend.mean(backend.square(x), axis=[1, 2], keepdims=True) + 1e-4)
        x *= (1.0 / x_std)
        return x


class NoiseModulation(Layer):

    def __init__(self, activation=None, **kwargs):
        self.activation = activation
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'activation': self.activation})
        return config

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel = self.add_weight(name='kernel', shape=[input_shape[3]], initializer='zeros', trainable=True)
        self.bias = self.add_weight(name='bias', shape=[input_shape[3]], initializer='zeros', trainable=True)

    def call(self, inputs, **kwargs):
        noise = tf.random.normal(shape=[tf.shape(inputs)[0], inputs.shape[1], inputs.shape[2], 1], dtype=self.kernel.dtype)
        kernel = tf.reshape(self.kernel, [1, 1, 1, -1])
        bias = tf.reshape(self.bias, [1, 1, 1, -1])
        output = inputs + (kernel * noise + bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


class Constant(Layer):

    def __init__(self, shape, initializer, **kwargs):
        self.shape = shape
        self.initializer = initializer
        super(Constant, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'shape': self.shape})
        config.update({'initializer': self.initializer})
        return config

    def build(self, input_shape):
        super(Constant, self).build(input_shape)
        self.kernel = self.add_weight(name='kernel', shape=self.shape, initializer=self.initializer, trainable=True)

    def call(self, inputs, **kwargs):
        output = tf.tile(self.kernel, [tf.shape(inputs)[0], 1, 1, 1])
        return output


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
