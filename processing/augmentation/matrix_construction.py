import tensorflow as tf


def apply_probability_mask(probability, parameters, mask_value):
    shape = tf.shape(parameters)
    mask = tf.random.uniform(shape, 0, 1) < probability
    mask_value = tf.broadcast_to(tf.convert_to_tensor(mask_value, dtype=parameters.dtype), shape)
    masked_parameters = tf.where(mask, parameters, mask_value)
    return masked_parameters


def convert_rows_to_matrix(*rows):
    rows = [[tf.convert_to_tensor(x, dtype=tf.float32) for x in r] for r in rows]
    batch_elems = [x for r in rows for x in r if x.shape.rank != 0]
    assert all(x.shape.rank == 1 for x in batch_elems)
    batch_size = tf.shape(batch_elems[0])[0] if len(batch_elems) else 1
    rows = [[tf.broadcast_to(x, [batch_size]) for x in r] for r in rows]
    return tf.transpose(rows, [2, 0, 1])


def construct_2d_translation_matrix(tx, ty):
    return convert_rows_to_matrix(
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    )


def construct_3d_translation_matrix(tx, ty, tz):
    return convert_rows_to_matrix(
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    )


def construct_2d_rotation_matrix(theta):
    return convert_rows_to_matrix(
        [tf.cos(theta), tf.sin(-theta), 0],
        [tf.sin(theta), tf.cos(theta), 0],
        [0, 0, 1]
    )


def construct_3d_rotation_matrix(v, theta):
    vx = v[..., 0]
    vy = v[..., 1]
    vz = v[..., 2]
    s = tf.sin(theta)
    c = tf.cos(theta)
    cc = 1 - c
    return convert_rows_to_matrix(
        [vx * vx * cc + c, vx * vy * cc - vz * s, vx * vz * cc + vy * s, 0],
        [vy * vx * cc + vz * s, vy * vy * cc + c, vy * vz * cc - vx * s, 0],
        [vz * vx * cc - vy * s, vz * vy * cc + vx * s, vz * vz * cc + c, 0],
        [0, 0, 0, 1]
    )


def construct_2d_scale_matrix(sx, sy):
    return convert_rows_to_matrix(
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    )


def construct_3d_scale_matrix(sx, sy, sz):
    return convert_rows_to_matrix(
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    )


def construct_inv_2d_translation_matrix(tx, ty):
    return construct_2d_translation_matrix(-tx, -ty)


def construct_inv_2d_rotation_matrix(theta):
    return construct_2d_rotation_matrix(-theta)


def construct_inv_2d_scale_matrix(sx, sy):
    return construct_2d_scale_matrix(1 / sx, 1 / sy)
