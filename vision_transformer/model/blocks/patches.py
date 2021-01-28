import tensorflow as tf
from matplotlib import pyplot as plt


class PatchMaker(tf.keras.layers.Layer):

    def __init__(self, patch_size):
        super(PatchMaker, self).__init__()
        self.patch_size = patch_size

    def call(self, images, *args, **kwargs):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1], padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    @staticmethod
    def visualize_patches(self, patches, patch_size, figsize):
        n = int(tf.sqrt(patches.shape[1]))
        plt.figure(figsize=figsize)
        for i, patch in enumerate(patches[0]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
            plt.imshow(patch_img.numpy().astype("uint8"))
            plt.axis("off")
