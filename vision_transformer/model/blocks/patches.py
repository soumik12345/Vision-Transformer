from math import sqrt
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
    def visualize_patches(patches, patch_size, figsize):
        n = int(sqrt(patches.shape[1]))
        plt.figure(figsize=figsize)
        for i, patch in enumerate(patches[0]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
            plt.imshow(patch_img.numpy().astype("uint8"))
            plt.axis("off")
        plt.show()


class PatchEmbedding(tf.keras.layers.Layer):

    def __init__(self, num_patches: int, projection_dim: int):
        super(PatchEmbedding, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch, *args, **kwargs):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        combined_embedding = self.projection(patch) + self.position_embedding(positions)
        return combined_embedding
