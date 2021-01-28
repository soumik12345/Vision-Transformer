from typing import List
import tensorflow as tf


class MLPBlock(tf.keras.layers.Layer):

    def __init__(self, hidden_units: List[int], dropout_rate: float):
        super(MLPBlock, self).__init__()
        self.mlp_layers = []
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


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(
            self, num_heads: int, projection_dimension: int,
            epsilon: float, attention_dropout: float, mlp_dropout: float):
        super(TransformerEncoder, self).__init__()
        hidden_units = [projection_dimension * 2, projection_dimension]
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dimension, dropout=attention_dropout
        )
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.mlp_block = MLPBlock(hidden_units=hidden_units, dropout_rate=mlp_dropout)

    def call(self, encoded_patches, *args, **kwargs):
        x1 = self.layer_norm_1(encoded_patches)
        attention_output = self.attention(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        x3 = self.layer_norm_2(x2)
        x3 = self.mlp_block(x3)
        outputs = tf.keras.layers.Add()([x3, x2])
        return outputs
