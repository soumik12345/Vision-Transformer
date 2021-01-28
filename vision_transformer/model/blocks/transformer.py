from typing import List
import tensorflow as tf


class MLPBlock(tf.keras.layers.Layer):

    def __init__(self, hidden_units: List[int], dropout_rate: float):
        super(MLPBlock, self).__init__()
        self.mlp_layers = [], []
        for units in hidden_units:
            self.mlp_layers.append((
                tf.keras.layers.Dense(units, activation=tf.nn.gelu),
                tf.keras.layers.Dropout(dropout_rate)
            ))

    def call(self, x, *args, **kwargs):
        for mlp_layer in self.mlp_layers:
            x = mlp_layer[0](x)
            x = mlp_layer[1](x)
        return x
