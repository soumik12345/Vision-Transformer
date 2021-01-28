from typing import List
import tensorflow as tf
from .transformer import MLPBlock


class ClassificationBlock(tf.keras.layers.Layer):

    def __init__(
            self, epsilon: float, hidden_units: List[int], n_classes: int,
            representation_dropout_rate: float, mlp_dropout_rate: float):
        super(ClassificationBlock, self).__init__()
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(representation_dropout_rate)
        self.mlp_block = MLPBlock(
            hidden_units=hidden_units, dropout_rate=mlp_dropout_rate
        )
        self.output_layer = tf.keras.layers.Dense(n_classes)

    def __call__(self, encoded_patches, *args, **kwargs):
        x = self.layer_norm(encoded_patches)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.mlp_block(x)
        output = self.output_layer(x)
        return output
