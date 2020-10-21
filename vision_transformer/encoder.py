import tensorflow as tf


class MultiHeadSelfAttention(tf.keras.layers.Layer):

    def __init__(self, embedding_dimension, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        assert embedding_dimension % num_heads == 0, 'Invalid Embedding Dimension'
        self.projection_dim = embedding_dimension // num_heads
        self.query_dense = tf.keras.layers.Dense(embedding_dimension)
        self.key_dense = tf.keras.layers.Dense(embedding_dimension)
        self.value_dense = tf.keras.layers.Dense(embedding_dimension)
        self.combine_heads = tf.keras.layers.Dense(embedding_dimension)

    @staticmethod
    def attention(query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embedding_dimension)
        )
        output = self.combine_heads(concat_attention)
        return output


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, embedding_dimension, num_heads, feed_forward_dimension, dropout=0.1, epsilon=1e-6):
        super(TransformerEncoder, self).__init__()
        self.attention = MultiHeadSelfAttention(embedding_dimension, num_heads)
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(
                feed_forward_dimension, activation="relu"
            ),
            tf.keras.layers.Dense(embedding_dimension),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training):
        attention_output = self.attention(inputs)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)
        feed_forward_output = self.feed_forward(out1)
        feed_forward_output = self.dropout2(feed_forward_output, training=training)
        return self.layernorm2(out1 + feed_forward_output)
