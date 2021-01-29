import os
import wandb
from typing import List
import tensorflow as tf
from datetime import datetime
import tensorflow_addons as tfa
from .dataloaders import TFDSLoader
from .model import VisionTransformer
from wandb.keras import WandbCallback


class Trainer:

    def __init__(self, project_name: str, experiment_name: str, wandb_api_key: str):
        self.train_dataset = None
        self.test_dataset = None
        self.optimizer = None
        self.model: tf.keras.Model = None
        self.training_history = None
        self.callbacks = []
        os.environ['WANDB_API_KEY'] = wandb_api_key
        wandb.init(
            project=project_name,
            name=experiment_name,
            sync_tensorboard=True
        )

    def build_dataset(self, dataset_name: str, buffer_size: int, batch_size: int):
        loader = TFDSLoader(disable_progress_bar=True)
        loader.load_data(dataset_name=dataset_name)
        self.train_dataset, self.test_dataset = loader.get_datasets(
            buffer_size=buffer_size, batch_size=batch_size
        )

    def build_model(
            self, input_shape: List[int], image_size: int, rotate_factor: float, zoom_factor: float,
            patch_size: int, n_transformer_blocks: int, num_heads: int,
            projection_dimension: int, epsilon: float, attention_dropout: float, mlp_dropout: float,
            hidden_units: List[int], n_classes: int, representation_dropout_rate: float):
        self.model = VisionTransformer(
            input_shape=input_shape, image_size=image_size, rotate_factor=rotate_factor, zoom_factor=zoom_factor,
            patch_size=patch_size, n_transformer_layers=n_transformer_blocks,  num_heads=num_heads,
            projection_dimension=projection_dimension, epsilon=epsilon, attention_dropout=attention_dropout,
            mlp_dropout=mlp_dropout, hidden_units=hidden_units, n_classes=n_classes, projection_dim=projection_dimension,
            representation_dropout_rate=representation_dropout_rate, num_patches=(image_size // patch_size) ** 2
        )

    def compile(self, learning_rate: float, weight_decay: float):
        self.optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
        self.model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name="top-3-accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")
            ],
        )

    def summarize(self):
        self.model.summary()

    def _checkpoint_callback(self, checkpoint_dir: str):
        self.callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_dir, monitor="val_accuracy",
                save_best_only=True, save_weights_only=True,
            )
        )

    def _tensorboard_callback(self):
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1,
                update_freq=50, write_images=True
            )
        )
        self.callbacks.append(WandbCallback())

    def train(self, epochs, checkpoint_dir):
        self._checkpoint_callback(checkpoint_dir=checkpoint_dir)
        self._tensorboard_callback()
        self.training_history = self.model.fit(
            self.train_dataset,
            validation_data=self.test_dataset,
            epochs=epochs, callbacks=self.callbacks
        )
