import tensorflow as tf
from .attention import MultiHeadedAttention
from .blocks import MLPBlock, ResidualBlock, NormalizationBlock


class TransformerEncoder(tf.keras.Model):

    def __init__(self, dimension, depth, heads, mlp_dimension):
        super(TransformerEncoder, self).__init__()
        layers = []
        for _ in range(depth):
            layers += [
                ResidualBlock(
                    NormalizationBlock(
                        dimension,
                        MultiHeadedAttention(
                            dimension, heads=heads
                        )
                    )
                ),
                ResidualBlock(
                    NormalizationBlock(
                        dimension,
                        MLPBlock(dimension, mlp_dimension)
                    )
                )
            ]
        self.layers = tf.keras.Sequential(layers)

    def call(self, inputs):
        return self.layers(inputs)
