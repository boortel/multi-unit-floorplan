import gc

import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
import numpy as np


class AutomaticWeightedLoss:
    def __init__(self, loss_functions, names, inds, dec, epochs, log_dir_path):
        self.loss_functions = loss_functions
        self.names = names
        self.inds = inds
        self.dec = dec
        # T-3 fix: reparameterize in log-space to prevent NaN when sigma→0
        self.log_vars = []
        self.epoch = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='epoch')
        self.epochs = epochs
        self.losses = [tf.Variable(name=name, dtype=tf.float32,
                                   initial_value=0.0, trainable=False) for name in self.names]
        for i in list(dict.fromkeys(inds)):
            self.log_vars.append(tf.Variable(name='LogVar_' + str(i), dtype=tf.float32,
                                             initial_value=0.0, trainable=True))
        # Backward-compatible alias: model.loss_sigmas = loss_function.sigmas still works
        self.sigmas = self.log_vars
        # T-6 fix: track step count for epoch-averaged loss logging
        self.step_count = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='step_count')

        file_writer = tf.summary.create_file_writer(log_dir_path)
        file_writer.set_as_default()

    def combined_loss(self):
        def loss_function(y_true, y_pred):
            loss_sum = 0
            for i in range(len(self.loss_functions)):
                loss = self.loss_functions[i](y_true, y_pred)
                if self.dec[i]:
                    loss *= tf.math.pow(20.0, -self.epoch/self.epochs)
                self.losses[i].assign_add(loss)
                # T-3 fix: log-space weighting — precision = exp(-log_var)
                precision = tf.exp(-self.log_vars[self.inds[i]])
                loss_sum += precision * loss + self.log_vars[self.inds[i]]
            # T-6 fix: track step count for epoch averaging
            self.step_count.assign_add(1.0)
            return loss_sum

        return loss_function

    def reset(self):
        for loss in self.losses:
            loss.assign(0.0)
        self.step_count.assign(0.0)


class AutomaticWeightedLossCallback(Callback):
    def __init__(self, aaf):
        self.aaf = aaf

    def on_epoch_end(self, epoch, logs=None):
        self.model.automatic_loss.epoch.assign(float(epoch))

        sigmas = []
        with tf.name_scope("Sigmas"):
            for sigma in self.model.loss_sigmas:
                val = sigma.numpy()
                sigmas.append(val)
                tf.summary.scalar(sigma.name, data=val, step=epoch)
            print('sigmas', sigmas)

        if self.aaf:
            w_edge_norm = tf.nn.softmax(self.model.w_edge, axis=-1).numpy()[0][0][0]
            w_not_edge_norm = tf.nn.softmax(self.model.w_not_edge, axis=-1).numpy()[0][0][0]
            print('w_edge_norm', w_edge_norm)
            print('w_not_edge_norm', w_not_edge_norm)
            print('w_edge_norm_mean', np.mean(w_edge_norm, axis=0)[0])
            print('w_not_edge_norm_mean', np.mean(w_not_edge_norm, axis=0)[0])

        with tf.name_scope("Losses"):
            step_count = max(self.model.automatic_loss.step_count.numpy(), 1.0)  # T-6 fix
            for i in range(len(self.model.automatic_loss.losses)):
                val = self.model.automatic_loss.losses[i].numpy()
                avg_val = val / step_count  # T-6 fix: log epoch average, not sum
                print(avg_val)
                tf.summary.scalar(self.model.automatic_loss.names[i], data=avg_val, step=epoch)
            self.model.automatic_loss.reset()
