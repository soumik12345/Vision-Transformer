from vision_transformer import Trainer


trainer = Trainer()
trainer.build_dataset(
    dataset_name='cifar100', buffer_size=1024, batch_size=16
)
trainer.build_model(
    input_shape=[32, 32, 3], image_size=72, rotate_factor=0.2, zoom_factor=0.2,
    patch_size=6, n_transformer_blocks=8, num_heads=4, projection_dimension=64,
    epsilon=1e-6, attention_dropout=0.1, mlp_dropout=0.1, hidden_units=[2048, 1024],
    n_classes=100, representation_dropout_rate=0.5
)
trainer.summarize()
trainer.compile(learning_rate=1e-3, weight_decay=1e-3)
trainer.train(epochs=100, checkpoint_dir='./checkpoints/')
