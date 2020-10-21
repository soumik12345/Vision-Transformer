import os
import wandb
import tensorflow as tf
from wandb.keras import WandbCallback
from tensorflow_addons.optimizers import AdamW
from vision_transformer import DataLoader, VisionTransformer


def train():
    wandb.init(project="vision-transformer")
    train_dataset, test_dataset = DataLoader().get_datasets(1024, 64)
    strategy = tf.distribute.MirroredStrategy() if len(
        tf.config.list_physical_devices('GPU')) > 1 else tf.distribute.OneDeviceStrategy("GPU:0")
    with strategy.scope():
        model = VisionTransformer({
            'image_size': 32, 'patch_size': 4,
            'num_classes': 10, 'num_layers': 3,
            'd_model': 64, 'num_heads': 4, 'mlp_dim': 128
        })
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=AdamW(learning_rate=1e-3, weight_decay=0.1), metrics=["accuracy"],
        )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(wandb.run.dir, "model.h5"),
            save_best_only=True, monitor='val_loss', mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=3, verbose=0,
            mode='auto', min_delta=0.0001, cooldown=0, min_lr=0
        ),
        WandbCallback()
    ]
    model.fit(
        train_dataset, validation_data=test_dataset,
        epochs=10, callbacks=callbacks
    )
