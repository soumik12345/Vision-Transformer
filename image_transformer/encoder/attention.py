import tensorflow as tf
from einops.layers.tensorflow import Rearrange


class MultiHeadedAttention(tf.keras.Model):

    def __init__(self, dimension, heads=8):
        super(MultiHeadedAttention, self).__init__()

        self.heads = heads
        self.scale = dimension ** -0.5

        self.mlp_in = tf.keras.layers.Dense(dimension * 3, use_bias=False)
        self.mlp_out = tf.keras.layers.Dense(dimension)

        self.rearrange_attention = Rearrange(
            'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        self.rearrange_output = Rearrange('b h n d -> b n (h d)')

    def call(self, inputs):
        query_key_value = self.mlp_in(inputs)
        query_key_value = self.rearrange_attention(query_key_value)

        query = query_key_value[0]
        key = query_key_value[1]
        value = query_key_value[2]

        dot_product = tf.einsum('bhid,bhjd->bhij', query, key) * self.scale
        attention = tf.nn.softmax(dot_product, axis=-1)

        output = tf.einsum('bhij,bhjd->bhid', attention, value)
        output = self.rearrange_output(output)
        return self.mlp_out(output)
