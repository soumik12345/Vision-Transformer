import tensorflow as tf
from echoAI.Activation.TF_Keras.custom_activation import GELU


class ResidualBlock(tf.keras.Model):

    def __init__(self, residual_function):
        super(ResidualBlock, self).__init__()
        self.residual_function = residual_function

    def call(self, inputs):
        return self.residual_function(inputs) + inputs


class NormalizationBlock(tf.keras.Model):

    def __init__(self, norm_function, epsilon=1e-5):
        super(NormalizationBlock, self).__init__()
        self.norm_function = norm_function
        self.normalize = tf.keras.layers.LayerNormalization(epsilon=epsilon)

    def call(self, inputs):
        return self.norm_function(self.normalize(inputs))


class MLPBlock(tf.keras.Model):

    def __init__(self, output_dimension, hidden_dimension):
        super(MLPBlock, self).__init__()
        self.mlp_1 = tf.keras.layers.Dense(hidden_dimension)
        self.mlp_2 = tf.keras.layers.Dense(output_dimension)
        self.activation = GELU()

    def call(self, inputs):
        y = self.mlp_1(inputs)
        y = self.activation(y)
        y = self.mlp_2(y)
        return self.activation(y)
