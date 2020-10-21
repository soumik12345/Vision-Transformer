import tensorflow as tf
from .encoder import TransformerEncoder
from tensorflow_addons.activations import gelu


class VisionTransformer(tf.keras.Model):

    def __init__(self, model_configs, channels=3, dropout=0.1):
        super(VisionTransformer, self).__init__()
        num_patches = (model_configs['image_size'] // model_configs['patch_size']) ** 2
        self.patch_dim = channels * model_configs['patch_size'] ** 2
        self.patch_size = model_configs['patch_size']
        self.d_model = model_configs['d_model']
        self.num_layers = model_configs['num_layers']
        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
        self.pos_emb = self.add_weight("pos_emb", shape=(1, num_patches + 1, self.d_model))
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, self.d_model))
        self.patch_proj = tf.keras.layers.Dense(self.d_model)
        self.enc_layers = []
        for _ in range(self.num_layers):
            self.enc_layers.append(
                TransformerEncoder(
                    self.d_model, model_configs['num_heads'],
                    model_configs['mlp_dim'], dropout
                )
            )
        self.mlp_head = tf.keras.Sequential([
            tf.keras.layers.Dense(model_configs['mlp_dim'], activation=gelu),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(model_configs['num_classes']),
        ])

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1], padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, inputs, training):
        batch_size = tf.shape(inputs)[0]
        inputs = self.rescale(inputs)
        patches = self.extract_patches(inputs)
        inputs = self.patch_proj(patches)
        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.d_model])
        inputs = tf.concat([class_emb, inputs], axis=1)
        inputs = inputs + self.pos_emb
        for layer in self.enc_layers:
            inputs = layer(inputs, training)
        # First (class token) is used for classification
        x = self.mlp_head(inputs[:, 0])
        return inputs
