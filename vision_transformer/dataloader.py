import tensorflow as tf
import tensorflow_datasets as tfds


class DataLoader:

    def __init__(self, dataset_name='cifar10'):
        self.dataset = tfds.load(dataset_name, as_supervised=True)

    def get_datasets(self, buffer_size, batch_size):
        auto_tune = tf.data.experimental.AUTOTUNE
        train_dataset = self.dataset['train']
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(buffer_size)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(auto_tune)
        test_dataset = self.dataset["test"]
        test_dataset = test_dataset.cache()
        test_dataset = test_dataset.batch(batch_size)
        test_dataset = test_dataset.prefetch(auto_tune)
        return train_dataset, test_dataset
