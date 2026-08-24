import logging
import math
from enum import Enum
from typing import Callable

import tensorflow as tf
import tensorflow.keras.backend as K

logger = logging.getLogger(__name__)


class SchedulerType(Enum):
    WARMUP_LINEAR_DECAY = "warmup-linear-decay"
    COSINE_DECAY = "cosine-decay"
    COSINE_DECAY_WITH_WARMUP = "cosine-decay-warmup"
    REDUCE_LR_ON_PLATEAU = "reduce-lr-on-plateau"


def get(scheduler: SchedulerType, train_dataset_size: int, learning_rate: float, **hyperparams):
    batch_size = hyperparams.get("batch_size", 1)
    epochs = hyperparams.get("epochs", 1)
    steps_per_epoch = max(1, (train_dataset_size + batch_size - 1) // batch_size)
    total_steps = steps_per_epoch * epochs

    if scheduler == SchedulerType.WARMUP_LINEAR_DECAY:
        warmup_steps = int(total_steps * hyperparams["warmup_proportion"])
        logger.info("Total steps %s, warmup steps %s", total_steps, warmup_steps)
        schedule = WarmupLinearDecaySchedule(warmup_steps, total_steps, learning_rate)
        return LearningRateScheduler(schedule, steps_per_epoch, verbose=0)

    elif scheduler == SchedulerType.COSINE_DECAY:
        min_lr = hyperparams.get("min_lr", 1e-6)
        logger.info("CosineDecay: total_steps=%s lr=%.2e -> %.2e", total_steps, learning_rate, min_lr)
        schedule = CosineDecaySchedule(total_steps, learning_rate, min_lr=min_lr)
        return LearningRateScheduler(schedule, steps_per_epoch, verbose=0)

    elif scheduler == SchedulerType.COSINE_DECAY_WITH_WARMUP:
        min_lr = hyperparams.get("min_lr", 1e-6)
        warmup_epochs = hyperparams.get("warmup_epochs", 5)
        warmup_steps = steps_per_epoch * warmup_epochs
        logger.info("CosineDecayWithWarmup: warmup=%d steps, total=%d steps, lr=%.2e -> %.2e",
                    warmup_steps, total_steps, learning_rate, min_lr)
        schedule = CosineDecayWithWarmupSchedule(total_steps, learning_rate,
                                                  warmup_steps=warmup_steps, min_lr=min_lr)
        return LearningRateScheduler(schedule, steps_per_epoch, verbose=0)

    else:
        raise ValueError("Unknown scheduler %s" % scheduler)


class LearningRateScheduler(tf.keras.callbacks.Callback):
    # Currently, the optimizers in TF2 don't properly support LR schedulers as callable.
    # As alternative we have to use a Keras callback which only allows for updating the LR per batch instead per step

    """Learning rate scheduler.
    Arguments:
        schedule: a function that takes an step index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule:Callable[[int], float], steps_per_epoch:int, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose
        self._current_step = 0

    def on_train_batch_begin(self, batch, logs=None):
        new_lr = self.schedule(self._current_step)

        K.set_value(self.model.optimizer.lr, new_lr)

        self._current_step += 1

        if self.verbose > 0:
            logger.info('\nBatch %05d: LearningRateScheduler changing learning rate to %s.', batch + 1, new_lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['learning_rate'] = K.get_value(self.model.optimizer.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['learning_rate'] = K.get_value(self.model.optimizer.lr)


class WarmupLinearDecaySchedule:
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, warmup_steps, total_steps, learning_rate, min_lr=0.0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_learning_rate = learning_rate
        self.min_lr = min_lr
        self.decay_steps = max(1.0, self.total_steps - self.warmup_steps)

    def __call__(self, step):
        if step < self.warmup_steps:
            learning_rate = self.initial_learning_rate * float(step) / max(1., self.warmup_steps)
        else:
            decay_factor = max(0, (self.total_steps - step) / self.decay_steps)
            learning_rate = self.min_lr + (self.initial_learning_rate - self.min_lr) * decay_factor

        return learning_rate


class CosineDecaySchedule:
    """Cosine annealing from `learning_rate` → `min_lr` over `total_steps` training steps.
    Follows: lr(t) = min_lr + 0.5*(lr - min_lr)*(1 + cos(pi * t / total_steps))
    """
    def __init__(self, total_steps, learning_rate, min_lr=1e-6):
        self.total_steps = max(1, total_steps)
        self.initial_learning_rate = learning_rate
        self.min_lr = min_lr

    def __call__(self, step):
        progress = min(float(step) / self.total_steps, 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.initial_learning_rate - self.min_lr) * cosine_factor


class CosineDecayWithWarmupSchedule:
    """Linear warmup then cosine annealing.
    Linearly increases LR from 0 to `learning_rate` over `warmup_steps`,
    then applies cosine decay from `learning_rate` → `min_lr` over the remaining steps.
    """
    def __init__(self, total_steps, learning_rate, warmup_steps=0, min_lr=1e-6):
        self.total_steps = max(1, total_steps)
        self.initial_learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.decay_steps = max(1, total_steps - warmup_steps)

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.initial_learning_rate * float(step) / max(1.0, float(self.warmup_steps))
        progress = min(float(step - self.warmup_steps) / self.decay_steps, 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.initial_learning_rate - self.min_lr) * cosine_factor

