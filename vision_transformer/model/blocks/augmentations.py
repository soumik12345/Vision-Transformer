import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing as tfpr


def Augmentations(image_size: int, rotate_factor: float, zoom_factor: float):
    return tf.keras.Sequential([
        tfpr.Normalization(),
        tfpr.Resizing(image_size, image_size),
        tfpr.RandomFlip('horizontal'),
        tfpr.RandomRotation(factor=rotate_factor),
        tfpr.RandomZoom(
            height_factor=zoom_factor, width_factor=zoom_factor
        )
    ], name='Data_Augmentation')
