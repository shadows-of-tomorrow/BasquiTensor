import numpy as np

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Layer, Add, Dense, Conv2D
from tensorflow.keras import backend


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

    def _merge_function(self, pixs):
        assert len(pixs) == 2
        return (1.0 - self.alpha) * pixs[0] + self.alpha * pixs[1]


class DenseEQL(Dense):

    def __init__(self, gain=np.sqrt(2), **kwargs):
        self.gain = gain
        super().__init__(kernel_initializer=RandomNormal(), **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        n_inputs = np.product([int(val) for val in input_shape[1:]])
        self.c = self.gain / np.sqrt(n_inputs)

    def call(self, inputs):
        output = backend.dot(inputs, self.kernel * self.c)
        if self.use_bias:
            output = backend.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


class Conv2DEQL(Conv2D):

    def __init__(self, gain=np.sqrt(2), **kwargs):
        self.gain = gain
        super().__init__(kernel_initializer=RandomNormal(), **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        n_inputs = np.product([int(val) for val in input_shape[1:]])
        self.c = self.gain / np.sqrt(n_inputs)

    def call(self, inputs):
        output = backend.conv2d(inputs, self.kernel * self.c, strides=self.strides, padding=self.padding,
                                data_format=self.data_format, dilation_rate=self.dilation_rate)
        if self.use_bias:
            output = backend.bias_add(output, self.bias, data_format=self.data_format)
        if self.activation is not None:
            return self.activation(output)
        return output
