import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from processing.augmentation.matrix_construction import apply_probability_mask
from processing.augmentation.matrix_construction import construct_2d_scale_matrix
from processing.augmentation.matrix_construction import construct_2d_translation_matrix
from processing.augmentation.matrix_construction import construct_inv_2d_scale_matrix
from processing.augmentation.matrix_construction import construct_inv_2d_rotation_matrix
from processing.augmentation.matrix_construction import construct_inv_2d_translation_matrix


WAVELET = [
    0.015404109327027373,
    0.0034907120842174702,
    -0.11799011114819057,
    -0.048311742585633,
    0.4910559419267466,
    0.787641141030194,
    0.3379294217276218,
    -0.07263752278646252,
    -0.021060292512300564,
    0.04472490177066578,
    0.0017677118642428036,
    -0.007800708325034148
]


class GeometryTransformer:

    def __init__(self, p_augment):
        # General augmentation probability.
        self.p_augment = p_augment
        # Geometric transformation probabilities.
        self.p_flip = 1.00
        self.p_90d_rotation = 1.00
        self.p_int_translation = 1.00
        self.p_iso_scaling = 1.00
        self.p_arb_rotation = 1.00
        self.p_ani_scaling = 1.00
        self.p_frac_translation = 1.00
        # Geometric transformation intensities.
        self.translate_max = 0.125
        self.iso_scaling_std = 0.20
        self.rotate_max = 1.00
        self.ani_scaling_std = 0.20
        self.frac_translation_std = 0.125

    def transform_tensors(self, x, batch_size, width, height, channels):
        g = self._construct_geometric_transforms(batch_size, width, height)
        x = self._apply_geometric_transforms(g, x, width, height, channels)
        return x

    def _apply_geometric_transforms(self, g_inv, x, width, height, channels):
        hz, hz_pad = self._construct_low_pass_filter(channels)
        pad, t_in, t_out = self._compute_padding_transform(g_inv, hz_pad, width, height)
        g_inv, x = self._apply_padding_transform(g_inv, x, pad, t_in, t_out)
        g_inv, x = self._apply_upsampling_transform(g_inv, x, hz)
        transforms = tf.reshape(g_inv, [-1, 9])[:, :8]
        shape = [(height + hz_pad * 2) * 2, (width + hz_pad * 2) * 2]
        x = tfa.image.transform(images=x, transforms=transforms, output_shape=shape, interpolation='BILINEAR')
        x = self.apply_downsampling(x, hz, hz_pad, width, height)
        return x

    def _construct_geometric_transforms(self, batch_size, width, height):
        g_inv = tf.eye(3, batch_shape=[batch_size])
        g_inv = self._multiply_x_flips(g_inv, batch_size)
        g_inv = self._multiply_90_degree_rotations(g_inv, batch_size)
        g_inv = self._multiply_integer_translations(g_inv, batch_size, width, height)
        g_inv = self._multiply_isotropic_scaling(g_inv, batch_size)
        g_inv, p_pre_rotation = self._multiply_pre_rotation(g_inv, batch_size)
        g_inv = self._multiply_anisotropic_scaling(g_inv, batch_size)
        g_inv = self._multiply_post_rotation(g_inv, p_pre_rotation, batch_size)
        g_inv = self._multiply_fractional_translation(g_inv, batch_size, width, height)
        return g_inv

    def _multiply_x_flips(self, g_inv, batch_size):
        n_flips = tf.floor(tf.random.uniform([batch_size], 0, 2))
        n_flips = apply_probability_mask(self.p_augment * self.p_flip, n_flips, 0)
        g_flip = construct_inv_2d_scale_matrix(1 - 2 * n_flips, 1)
        return g_inv @ g_flip

    def _multiply_90_degree_rotations(self, g_inv, batch_size):
        n_rotations = tf.floor(tf.random.uniform([batch_size], 0, 4))
        n_rotations = apply_probability_mask(self.p_augment * self.p_90d_rotation, n_rotations, 0)
        g_90d_rotation = construct_inv_2d_rotation_matrix(-np.pi / 2 * n_rotations)
        return g_inv @ g_90d_rotation

    def _multiply_integer_translations(self, g_inv, batch_size, width, height):
        int_translate = tf.random.uniform([batch_size, 2], -self.translate_max, self.translate_max)
        int_translate = apply_probability_mask(self.p_augment * self.p_int_translation, int_translate, 0)
        tx = tf.math.rint(int_translate[:, 0] * width)
        ty = tf.math.rint(int_translate[:, 1] * height)
        g_integer_translation = construct_inv_2d_translation_matrix(tx, ty)
        return g_inv @ g_integer_translation

    def _multiply_isotropic_scaling(self, g_inv, batch_size):
        iso_scaling = 2 ** tf.random.normal([batch_size], 0, self.iso_scaling_std)
        iso_scaling = apply_probability_mask(self.p_augment * self.p_iso_scaling, iso_scaling, 1)
        g_iso_scaling = construct_inv_2d_scale_matrix(iso_scaling, iso_scaling)
        return g_inv @ g_iso_scaling

    def _multiply_pre_rotation(self, g_inv, batch_size):
        p_pre_rotation = 1 - tf.sqrt(tf.cast(tf.maximum(1 - self.p_augment * self.p_arb_rotation, 0), tf.float32))
        pre_rotation = tf.random.uniform([batch_size], -np.pi * self.rotate_max, np.pi * self.rotate_max)
        pre_rotation = apply_probability_mask(p_pre_rotation, pre_rotation, 0)
        g_pre_rotation = construct_inv_2d_rotation_matrix(-pre_rotation)
        return g_inv @ g_pre_rotation, p_pre_rotation

    def _multiply_anisotropic_scaling(self, g_inv, batch_size):
        ani_scaling = 2 ** tf.random.normal([batch_size], 0, self.ani_scaling_std)
        ani_scaling = apply_probability_mask(self.p_augment * self.p_ani_scaling, ani_scaling, 1)
        g_ani_scaling = construct_inv_2d_scale_matrix(ani_scaling, 1 / ani_scaling)
        return g_inv @ g_ani_scaling

    def _multiply_post_rotation(self, g_inv, p_pre_rotation, batch_size):
        p_post_rotation = p_pre_rotation
        post_rotation = tf.random.uniform([batch_size], -np.pi * self.rotate_max, np.pi * self.rotate_max)
        post_rotation = apply_probability_mask(p_post_rotation, post_rotation, 0)
        g_post_rotation = construct_inv_2d_rotation_matrix(-post_rotation)
        return g_inv @ g_post_rotation

    def _multiply_fractional_translation(self, g_inv, batch_size, width, height):
        frac_translation = tf.random.normal([batch_size, 2], 0, self.frac_translation_std)
        frac_translation = apply_probability_mask(self.p_augment * self.p_frac_translation, frac_translation, 0)
        tx = frac_translation[:, 0] * width
        ty = frac_translation[:, 1] * height
        g_frac_translation = construct_inv_2d_translation_matrix(tx, ty)
        return g_inv @ g_frac_translation

    def _construct_low_pass_filter(self, channels):
        hz = WAVELET
        hz = np.asarray(hz, dtype=np.float32)
        hz = np.reshape(hz, [-1, 1, 1]).repeat(channels, axis=1)
        hz_pad = hz.shape[0] // 4
        return hz, hz_pad

    def _compute_padding_transform(self, t, hz_pad, width, height):
        cx = (width - 1) / 2
        cy = (height - 1) / 2
        cp = np.transpose([[-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1]])
        cp = t @ cp[np.newaxis]
        cp = cp[:, :2, :]
        m_lo = tf.math.ceil(tf.reduce_max(-cp, axis=[0, 2]) - [cx, cy] + hz_pad * 2)
        m_hi = tf.math.ceil(tf.reduce_max(cp, axis=[0, 2]) - [cx, cy] + hz_pad * 2)
        m_lo = tf.clip_by_value(m_lo, [0, 0], [width - 1, height - 1])
        m_hi = tf.clip_by_value(m_hi, [0, 0], [width - 1, height - 1])
        pad = [[0, 0], [m_lo[1], m_hi[1]], [m_lo[0], m_hi[0]], [0, 0]]
        t_in = construct_2d_translation_matrix(cx + m_lo[0], cy + m_lo[1])
        t_out = construct_inv_2d_translation_matrix(cx + hz_pad, cy + hz_pad)
        return pad, t_in, t_out

    def _apply_padding_transform(self, t, x, pad, t_in, t_out):
        x = tf.pad(tensor=x, paddings=pad, mode='REFLECT')
        t = t_in @ t @ t_out
        return t, x

    def _apply_upsampling_transform(self, t, x, hz):
        shape = [tf.shape(x)[0], tf.shape(x)[1] * 2, tf.shape(x)[2] * 2, tf.shape(x)[3]]
        x = tf.nn.depthwise_conv2d_backprop_input(
            input_sizes=shape,
            filter=hz[np.newaxis, :],
            out_backprop=x,
            strides=[1, 2, 2, 1],
            padding='SAME',
            data_format='NHWC'
        )
        x = tf.nn.depthwise_conv2d_backprop_input(
            input_sizes=shape,
            filter=hz[:, np.newaxis],
            out_backprop=x,
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        )
        s_in = construct_2d_scale_matrix(2, 2)
        s_out = construct_inv_2d_scale_matrix(2, 2)
        t = s_in @ t @ s_out
        return t, x

    def apply_downsampling(self, x, hz, hz_pad, width, height):
        x = tf.nn.depthwise_conv2d(
            input=x,
            filter=hz[np.newaxis, :],
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        )
        x = tf.nn.depthwise_conv2d(
            input=x,
            filter=hz[:, np.newaxis],
            strides=[1, 2, 2, 1],
            padding='SAME',
            data_format='NHWC'
        )
        x = x[:, hz_pad: height + hz_pad, hz_pad: width + hz_pad, :]
        return x