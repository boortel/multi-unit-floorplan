import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
import numpy as np
#from tensorflow.python.keras.engine.training import _minimize
#from tensorflow.python.keras.mixed_precision.experimental.loss_scale_optimizer import LossScaleOptimizer
# For tensorflow 2.4.0
from tensorflow.python.keras.mixed_precision.loss_scale_optimizer import LossScaleOptimizer


class BaseModel(Model):
    def __init__(self, *args, tta=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.tta = tta

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        optimizer = self.optimizer

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
            if hasattr(optimizer, 'get_scaled_loss'):
                loss = optimizer.get_scaled_loss(loss)

        trainable_variables = self.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)
        if hasattr(optimizer, 'get_unscaled_gradients'):
            gradients = optimizer.get_unscaled_gradients(gradients)

        # Negate gradients for AAF edge variables by name, not position (T-2 fix)
        gradients = list(gradients)
        for i, v in enumerate(trainable_variables):
            if 'edge' in v.name and gradients[i] is not None:
                gradients[i] = -gradients[i]

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        # self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """The logic for one evaluation step.
        This method can be overridden to support custom evaluation logic.
        This method is called by `Model.make_test_function`.
        This function should contain the mathematical logic for one step of
        evaluation.
        This typically includes the forward pass, loss calculation, and metrics
        updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_test_function`, which can also be overridden.
        Args:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        if self.tta:
            image = x.numpy()[0]
            image_batch = np.array([
                image,
                np.fliplr(image),
                np.flipud(image),
                np.rot90(image, k=1),
                np.rot90(image, k=3),
            ])
            prediction = self(image_batch, training=False).numpy()
            # Average softmax probabilities, not discrete indices (E-1 fix)
            results = [
                prediction[0],
                np.fliplr(prediction[1]),
                np.flipud(prediction[2]),
                np.rot90(prediction[3], k=-1, axes=(0, 1)),
                np.rot90(prediction[4], k=-3, axes=(0, 1)),
            ]
            avg_prob = np.mean(results, axis=0)
            result = avg_prob.argmax(axis=-1).astype(np.uint8)
            # Use num_classes from model output, not image channels (E-2 fix)
            depth = prediction.shape[-1]
            y_pred = tf.one_hot(np.expand_dims(result, axis=0), depth=depth)
        else:
            y_pred = self(x, training=False)

        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}
