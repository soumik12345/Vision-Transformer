from typing import List
import tensorflow as tf
from .blocks import (
    Augmentations,
    PatchMaker, PatchEmbedding,
    TransformerEncoder, ClassificationBlock
)


def VisionTransformer(
        input_shape: List[int], image_size: int, rotate_factor: float, zoom_factor: float,
        patch_size: int, num_patches: int, projection_dim: int, n_transformer_layers: int,
        num_heads: int, projection_dimension: int, epsilon: float, attention_dropout: float,
        mlp_dropout: float, hidden_units: List[int], n_classes: int, representation_dropout_rate: float):
    input_tensor = tf.keras.Input(shape=input_shape, name='Input_Tensor')
    augmented_inputs = Augmentations(
        image_size=image_size,
        rotate_factor=rotate_factor,
        zoom_factor=zoom_factor
    )(input_tensor)
    patches = PatchMaker(patch_size=patch_size)(augmented_inputs)
    encoded_patches = PatchEmbedding(num_patches, projection_dim)(patches)
    for _ in range(n_transformer_layers):
        encoded_patches = TransformerEncoder(
            num_heads=num_heads, projection_dimension=projection_dimension,
            epsilon=epsilon, attention_dropout=attention_dropout, mlp_dropout=mlp_dropout
        )(encoded_patches)
    logits = ClassificationBlock(
        epsilon=epsilon, hidden_units=hidden_units, n_classes=n_classes,
        representation_dropout_rate=representation_dropout_rate, mlp_dropout_rate=mlp_dropout
    )(encoded_patches)
    model = tf.keras.Model(input_tensor, logits)
    return model
