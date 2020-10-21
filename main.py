import tensorflow as tf
from vision_transformer import KerasDataset, VisionTransformer


train_dataset, test_dataset = KerasDataset(tf.keras.datasets.cifar10).get_datasets()
strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices('GPU')) > 1 else tf.distribute.OneDeviceStrategy("GPU:0")
with strategy.scope():
    model = VisionTransformer(
        image_size=32, patch_size=4, n_classes=10, batch_size=64,
        dimension=64, depth=3, heads=4, mlp_dimension=128
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    train_distributed_dataset = strategy.experimental_distribute_dataset(train_dataset)
    model.compile(optimizer=optimizer, loss_fn=cross_entropy_loss)
history = model.fit(train_dataset, batch_size=64, epochs=10)
