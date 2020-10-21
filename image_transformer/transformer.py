import tensorflow as tf
from .encoder import TransformerEncoder
from einops.layers.tensorflow import Rearrange
from echoAI.Activation.TF_Keras.custom_activation import GELU


class ImageTransformer(tf.keras.Model):

    def __init__(
            self, image_size, patch_size, n_classes, batch_size,
            dimension, depth, heads, mlp_dimension, channels=3):
        super(ImageTransformer, self).__init__()
        assert image_size % patch_size == 0, 'invalid patch size for image size'

        num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.dimension = dimension
        self.batch_size = batch_size

        self.positional_embedding = self.add_weight(
            "position_embeddings", shape=[num_patches + 1, dimension],
            initializer=tf.keras.initializers.RandomNormal(), dtype=tf.float32
        )
        self.embedding_mlp = tf.keras.layers.Dense(dimension)
        self.classification_token = self.add_weight(
            "classification_token", shape=[1, 1, dimension],
            initializer=tf.keras.initializers.RandomNormal(), dtype=tf.float32
        )

        self.rearrange = Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=self.patch_size, p2=self.patch_size
        )
        self.transformer = TransformerEncoder(dimension, depth, heads, mlp_dimension)
        self.classification_identity = tf.identity
        self.mlp_1 = tf.keras.layers.Dense(mlp_dimension)
        self.gelu = GELU()
        self.output = tf.keras.layers.Dense(n_classes)

    def call(self, inputs):
        shapes = tf.shape(inputs)
        y = self.rearrange(inputs)
        y = self.embedding_mlp(y)
        cls_tokens = tf.broadcast_to(
            self.classification_token,
            (shapes[0], 1, self.dimension)
        )
        y = tf.concat((cls_tokens, inputs), axis=1)
        y += self.positional_embedding
        y = self.transformer(y)
        y = self.classification_identity(y[:, 0])
        y = self.mlp_1(y)
        y = self.gelu(y)
        return self.output(y)

    def compile(self, loss_fn, **kwargs):
        super(ImageTransformer, self).compile(**kwargs)
        self.loss_fn = loss_fn
        self.train_accuracy = tf.keras.metrics.Accuracy('training_accuracy', dtype=tf.float32)
        # self.test_accuracy = tf.keras.metrics.Accuracy('testing_accuracy', dtype=tf.float32)

    @tf.function
    def train_step(self, data):
        def _step(inputs):
            X, Y = inputs
            with tf.GradientTape() as tape:
                logits = self(X, training=True)
                num_labels = tf.shape(logits)[-1]
                label_mask = tf.math.logical_not(Y < 0)
                label_mask = tf.reshape(label_mask, (-1,))
                logits = tf.reshape(logits, (-1, num_labels))
                logits_masked = tf.boolean_mask(logits, label_mask)
                label_ids = tf.reshape(Y, (-1,))
                label_ids_masked = tf.boolean_mask(label_ids, label_mask)
                cross_entropy = self.loss_fn(label_ids_masked, logits_masked)
                loss = tf.reduce_sum(cross_entropy) * (1.0 / self.batch_size)
                y_pred = tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
                self.train_accuracy.update_state(tf.squeeze(Y), y_pred)
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(
                list(zip(grads, self.trainable_variables)))
            return cross_entropy

        total_loss = self.distribute_strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            self.distribute_strategy.run(_step, args=(data,)), axis=0
        )
        mean_loss = total_loss / self.batch_size
        return {
            'total_loss': total_loss,
            'mean_loss': mean_loss,
            'train_accuracy': self.train_accuracy.result()
        }
