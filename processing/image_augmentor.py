import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

SYM6_WAVELET = [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633,
                0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252,
                -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148]


class ImageAugmentor:

    def __init__(self, p_augment=0.75, p_flip=1.00, p_rotation=1.00, p_int=1.00):
        # Augmentation probability.
        self.p_augment = p_augment
        # Pixel blitting.
        self.p_flip = p_flip
        self.p_rotation = p_rotation
        self.p_int = p_int
        self.x_int_max = 0.125

    def run(self, x, is_tensor=True):
        # 1. Cast numpy array to tensor if needed.
        if not is_tensor:
            x = tf.convert_to_tensor(value=x, dtype='float32')
        # 2. Get image dimensions.
        batch_size, width, height, channels = x.shape.as_list()
        # 3. Construct (inverse) transformation matrix.
        t_inv = self._construct_transform_matrix(batch_size, width, height)
        # 4. Apply (inverse) transformation matrix.
        x = self._apply_transform_matrix(t_inv, x, width, height, channels)
        # 5. Cast tensor to numpy array if needed.
        if not is_tensor:
            x = np.asarray(x)
        return x

    def _apply_transform_matrix(self, t, x, width, height, channels):
        # 1. Setup orthogonal low-pass filter.
        hz = SYM6_WAVELET
        hz = np.asarray(hz, dtype=np.float32)
        hz = np.reshape(hz, [-1, 1, 1]).repeat(channels, axis=1)
        hz_pad = hz.shape[0] // 4
        # 2. Calculate padding.
        cx = (x.shape[1] - 1) / 2
        cy = (x.shape[2] - 1) / 2
        cp = np.transpose([[-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1]])
        cp = t @ cp[np.newaxis]
        cp = cp[:, :2, :]
        m_lo = tf.math.ceil(tf.reduce_max(-cp, axis=[0, 2]) - [cx, cy] + hz_pad * 2)
        m_hi = tf.math.ceil(tf.reduce_max(cp, axis=[0, 2]) - [cx, cy] + hz_pad * 2)
        m_lo = tf.clip_by_value(m_lo, [0, 0], [x.shape[1] - 1, x.shape[2] - 1])
        m_hi = tf.clip_by_value(m_hi, [0, 0], [x.shape[1] - 1, x.shape[2] - 1])
        # 3. Pad image and adjust origin.
        pad = [[0, 0], [m_lo[1], m_hi[1]], [m_lo[0], m_hi[0]], [0, 0]]
        x = tf.pad(tensor=x, paddings=pad, mode='REFLECT')
        t_in = self._construct_2d_translation_matrix(cx + m_lo[0], cy + m_lo[1])
        t_out = self._construct_inv_2d_translation_matrix(cx + hz_pad, cy + hz_pad)
        t = t_in @ t @ t_out
        # 4. Upsample to account for increase resolution.
        shape = [tf.shape(x)[0], tf.shape(x)[1] * 2, tf.shape(x)[2] * 2, channels]
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
        s_in = self._construct_2d_scale_matrix(2, 2)
        s_out = self._construct_inv_2d_scale_matrix(2, 2)
        t = s_in @ t @ s_out
        # 5. Apply transformation matrix.
        transforms = tf.reshape(t, [-1, 9])[:, :8]
        shape = [(height + hz_pad * 2) * 2, (width + hz_pad * 2) * 2]
        x = tfa.image.transform(
            images=x,
            transforms=transforms,
            output_shape=shape,
            interpolation='BILINEAR'
        )
        # 6. Downsample and crop.
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

    def _construct_transform_matrix(self, batch_size, width, height):
        # Initialize (inverse) transformation matrix: t_inv @ pixel_out = pixel_in.
        t_inv = tf.eye(3, batch_shape=[batch_size])
        # Multiply transformation matrix with x-flips.
        t_inv = self._multiply_x_flips(t_inv, batch_size)
        # Multiply transformation matrix with 90 degree rotations.
        t_inv = self._multiply_90_degree_rotations(t_inv, batch_size)
        # Multiply transformation matrix with integer translations.
        t_inv = self._multiply_integer_translation(t_inv, batch_size, width, height)
        return t_inv

    def _multiply_x_flips(self, t_inv, batch_size):
        # 1. Sample number x-axis flips, k=0,1.
        n_flips = tf.floor(tf.random.uniform([batch_size], 0, 2))
        # 2. Apply probability mask to number of x-axis flips, identity -> 0 flips.
        n_flips = self._mask_parameters(self.p_augment * self.p_flip, n_flips, 0)
        # 3. Convert number of flips to transformation matrix form.
        t_flip = self._construct_inv_2d_scale_matrix(1 - 2 * n_flips, 1)
        return t_inv @ t_flip

    def _multiply_90_degree_rotations(self, t_inv, batch_size):
        # 1. Sample number of 90 degree rotations, k=0,1,2,3.
        n_rotations = tf.floor(tf.random.uniform([batch_size], 0, 4))
        # 2. Apply probability mask to number of rotations, identity -> 0 rotations.
        n_rotations = self._mask_parameters(self.p_augment * self.p_rotation, n_rotations, 0)
        # 3. Convert number of rotations to transformation matrix form.
        t_90d_rotation = self._construct_inv_2d_rotation_matrix(-np.pi / 2 * n_rotations)
        return t_inv @ t_90d_rotation

    def _multiply_integer_translation(self, t_inv, batch_size, width, height):
        # 1. Sample integer translation intensity.
        x_int = tf.random.uniform([batch_size, 2], -self.x_int_max, self.x_int_max)
        # 2. Apply probability mask to integer translation intensity, identity -> 0 intensity.
        x_int = self._mask_parameters(self.p_augment * self.p_int, x_int, 0)
        # 3. Convert integer translation intensity to transformation matrix form.
        tx = tf.math.rint(x_int[:, 0] * width)
        ty = tf.math.rint(x_int[:, 1] * height)
        t_integer_translation = self._construct_inv_2d_translation_matrix(tx, ty)
        return t_inv @ t_integer_translation

    def _mask_parameters(self, probability, parameters, mask_value):
        # 1. Generate mask: Parameter = identity if masked.
        shape = tf.shape(parameters)
        mask = tf.random.uniform(shape, 0, 1) < probability
        # 2. Broadcast and apply mask.
        mask_value = tf.broadcast_to(tf.convert_to_tensor(mask_value, dtype=parameters.dtype), shape)
        masked_parameters = tf.where(mask, parameters, mask_value)
        return masked_parameters

    def _convert_rows_to_matrix(self, *rows):
        rows = [[tf.convert_to_tensor(x, dtype=tf.float32) for x in r] for r in rows]
        batch_elems = [x for r in rows for x in r if x.shape.rank != 0]
        assert all(x.shape.rank == 1 for x in batch_elems)
        batch_size = tf.shape(batch_elems[0])[0] if len(batch_elems) else 1
        rows = [[tf.broadcast_to(x, [batch_size]) for x in r] for r in rows]
        return tf.transpose(rows, [2, 0, 1])

    def _construct_2d_translation_matrix(self, tx, ty):
        return self._convert_rows_to_matrix(
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        )

    def _construct_2d_rotation_matrix(self, theta):
        return self._convert_rows_to_matrix(
            [tf.cos(theta), tf.sin(-theta), 0],
            [tf.sin(theta), tf.cos(theta), 0],
            [0, 0, 1]
        )

    def _construct_2d_scale_matrix(self, sx, sy):
        return self._convert_rows_to_matrix(
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        )

    def _construct_inv_2d_translation_matrix(self, tx, ty):
        return self._construct_2d_translation_matrix(-tx, -ty)

    def _construct_inv_2d_rotation_matrix(self, theta):
        return self._construct_2d_rotation_matrix(-theta)

    def _construct_inv_2d_scale_matrix(self, sx, sy):
        return self._construct_2d_scale_matrix(1 / sx, 1 / sy)
