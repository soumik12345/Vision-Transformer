import tensorflow as tf


class KerasDataset:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        (
            self.x_train, self.y_train,
            self.x_test, self.y_test
        ) = self.load_data()

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = self.dataset_name.load_data()
        return x_train, y_train, x_test, y_test

    def get_shape(self):
        return self.x_train.shape[1:], self.y_train.shape[1:]

    def preprocess(self):
        if self.dataset_name == tf.keras.datasets.mnist:
            (height, width), _ = self.get_shape()
            self.x_train = tf.cast(
                self.x_train.reshape((-1, height, width)), dtype=tf.float32) / 255.
            self.x_test = tf.cast(
                self.x_test.reshape((-1, height, width)), dtype=tf.float32) / 255.
        else:
            (height, width, channels), _ = self.get_shape()
            self.x_train = tf.cast(
                self.x_train.reshape((-1, channels, height, width)), dtype=tf.float32) / 255.
            self.x_test = tf.cast(
                self.x_test.reshape((-1, channels, height, width)), dtype=tf.float32) / 255.

    def get_datasets(self):
        self.preprocess()
        return (
            tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(self.x_train),
                tf.data.Dataset.from_tensor_slices(self.y_train)
            )),
            tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(self.x_test),
                tf.data.Dataset.from_tensor_slices(self.y_test)
            ))
        )
