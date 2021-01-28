from typing import List
import tensorflow as tf
from .blocks import (
    Augmentations,
    PatchMaker, PatchEmbedding,
    TransformerEncoder, MLPBlock,
    ClassificationBlock
)


class VisionTransformer(tf.keras.Model):

    def __init__(
            self, image_size: int, rotate_factor: float, zoom_factor: float, patch_size: int, n_transformer_blocks: int,
            num_heads: int, projection_dimension: int, epsilon: float, attention_dropout: float, mlp_dropout: float,
            hidden_units: List[int], n_classes: int, representation_dropout_rate: float, mlp_dropout_rate: float):
        super(VisionTransformer, self).__init__()
        self.augment = Augmentations(
            image_size=image_size,
            rotate_factor=rotate_factor,
            zoom_factor=zoom_factor
        )
        self.make_patches = PatchMaker(patch_size=patch_size)
        self.patch_embeddings = PatchEmbedding(
            num_patches=(image_size // patch_size) ** 2, projection_dim=64
        )
        self.transformer_layers = []
        for _ in range(n_transformer_blocks):
            self.transformer_layers.append(
                TransformerEncoder(
                    num_heads=num_heads, projection_dimension=projection_dimension,
                    epsilon=epsilon, attention_dropout=attention_dropout, mlp_dropout=mlp_dropout
                )
            )
        self.classify = ClassificationBlock(
            epsilon=epsilon, hidden_units=hidden_units, n_classes=n_classes,
            representation_dropout_rate=representation_dropout_rate, mlp_dropout_rate=mlp_dropout_rate
        )

    def call(self, inputs, *args, **kwargs):
        augmented_inputs = self.augment(inputs)
        patches = self.make_patches(augmented_inputs)
        encoded_patches = self.patch_embeddings(patches)
        for transformer_layers in self.transformer_layers:
            encoded_patches = transformer_layers(encoded_patches)
        logits = self.classify(encoded_patches)
        return logits
