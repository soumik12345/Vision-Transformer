import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing as tfpr


class Augmentations(tf.keras.layers.Layer):

    def __init__(self, image_size: int, rotate_factor: float, zoom_factor: float):
        super(Augmentations, self).__init__()
        self.normalize = tfpr.Normalization()
        self.resize = tfpr.Resizing(image_size, image_size)
        self.mirror = tfpr.RandomFlip('horizontal')
        self.rotate = tfpr.RandomRotation(factor=rotate_factor)
        self.zoom = tfpr.RandomZoom(
            height_factor=zoom_factor, width_factor=zoom_factor
        )

    def call(self, inputs, *args, **kwargs):
        x = self.normalize(inputs)
        x = self.resize(x)
        x = self.mirror(x)
        x = self.rotate(x)
        x = self.zoom(x)
        return x
