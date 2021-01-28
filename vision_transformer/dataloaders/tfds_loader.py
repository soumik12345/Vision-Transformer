import tensorflow as tf
from typing import Tuple
import tensorflow_datasets as tfds
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset


AUTOTUNE = tf.data.experimental.AUTOTUNE


def configure_datasets(
        dataset: PrefetchDataset, buffer_size: int,
        batch_size: int, apply_shuffle: bool) -> PrefetchDataset:
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=buffer_size) if apply_shuffle else dataset
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


class TFDSLoader:

    def __init__(self, disable_progress_bar: bool = True):
        self.train_dataset = None
        self.test_dataset = None
        if disable_progress_bar:
            tfds.disable_progress_bar()

    def load_data(self, dataset_name: str):
        dataset = tfds.load(dataset_name, as_supervised=True)
        self.train_dataset = dataset['train']
        self.test_dataset = dataset['test']

    def get_datasets(self, buffer_size: int, batch_size: int) -> Tuple[PrefetchDataset, PrefetchDataset]:
        return (
            configure_datasets(
                self.train_dataset, buffer_size=1024,
                batch_size=16, apply_shuffle=True
            ),
            configure_datasets(
                self.test_dataset, buffer_size=1024,
                batch_size=16, apply_shuffle=False
            )
        )
