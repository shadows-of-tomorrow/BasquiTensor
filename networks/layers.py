import numpy as np
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Layer, Add, Dense, Conv2D
from tensorflow.keras import backend


class DenseEQL(Dense):
    """ Dense layer with equalized learning rate. """
    def __init__(self, **kwargs):
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel initializer.")
        super().__init__(kernel_initializer=RandomNormal(), **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        n_inputs = np.product([int(val) for val in input_shape[1:]])
        self.c = np.sqrt(2) / np.sqrt(n_inputs)

    def call(self, inputs):
        output = backend.dot(inputs, self.kernel * self.c)
        if self.use_bias:
            output = backend.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


class Conv2DEQL(Conv2D):
    """ Convolutional layer with equalized learning rate. """
    def __init__(self, **kwargs):
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel initializer.")
        super().__init__(kernel_initializer=RandomNormal(), **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        n_inputs = np.product([int(val) for val in input_shape[1:]])
        self.c = np.sqrt(2.0) / np.sqrt(n_inputs)

    def call(self, inputs):
        if self.rank == 2:
            outputs = backend.conv2d(
                inputs,
                self.kernel * self.c,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate
            )
        if self.use_bias:
            outputs = backend.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format
            )
        if self.activation is not None:
            return self.activation(outputs)


class PixelNormalization(Layer):
    """ Normalizes a tensor using a per-channel mean deviation. """
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    def call(self, pix, **kwargs):
        # 1. Square all pixel values.
        pix_sqr = pix ** 2.0
        # 2. Compute (channel-wise) mean squared pixel values.
        pix_sqr_mean = backend.mean(pix_sqr, axis=-1, keepdims=True)
        # 3. Offset mean values by a small number (prevent div. by zero error).
        pix_sqr_mean += 1.0e-8
        # 4. Compute normalization constant.
        pix_norm_const = backend.sqrt(pix_sqr_mean)
        # 5. Return normalized pixel values.
        return pix / pix_norm_const

    def compute_output_shape(self, input_shape):
        return input_shape


class MinibatchStDev(Layer):
    """ Appends a feature map to a pixel tensor. """
    def __init__(self, **kwargs):
        super(MinibatchStDev, self).__init__(**kwargs)

    def call(self, pix, **kwargs):
        # 1. Compute (example-wise) mean pixel values.
        pix_mean = backend.mean(pix, axis=0, keepdims=True)
        # 2. Compute squared difference between pixels and mean.
        pix_sqr_diff = backend.square(pix - pix_mean)
        # 3. Compute (example-wise) feature variances.
        pix_var = backend.mean(pix_sqr_diff, axis=0, keepdims=True)
        # 4. Offset mean values a small number (prevent div. by zero error).
        pix_var += 1e-08
        # 5. Compute (example-wise) feature standard deviations.
        pix_stdev = backend.sqrt(pix_var)
        # 6. Compute average of standard deviations.
        pix_mean_stdev = backend.mean(pix_stdev, keepdims=True)
        # 7. Create extra (constant) feature map.
        shape = backend.shape(pix)
        fmap_extra = backend.tile(pix_mean_stdev, (shape[0], shape[1], shape[2], 1))
        # 8. Concatenate extra feature map to input tensor (along channel-dim).
        pix_extra = backend.concatenate([pix, fmap_extra], axis=-1)
        return pix_extra

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1  # One additional feature map is added (channel dim +1).
        return tuple(input_shape)


class WeightedSum(Add):
    """ Computes the weighted sum of two pixel tensors. """

    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')

    def _merge_function(self, pixs):
        assert len(pixs) == 2
        return (1.0 - self.alpha) * pixs[0] + self.alpha * pixs[1]