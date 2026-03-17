# =========================================================
# Import all dependencies
# =========================================================
import numpy as np
import tensorflow as tf


def get_teacher_temp_schedule(epoch, warmup_epochs=30, start_temp=0.04, base_temp=0.07):
    """Linearly warms up teacher temperature during the first `warmup_epochs`."""
    if epoch < warmup_epochs:
        return start_temp + (base_temp - start_temp) * (epoch / warmup_epochs)
    return base_temp


def get_momentum_schedule(epoch, total_epochs, base_m=0.996, final_m=1.0):
    """Cosine decay schedule from base_m to final_m."""
    return final_m - (final_m - base_m) * 0.5 * (1. + np.cos(np.pi * epoch / total_epochs))

def warmup_schedule(epoch):
  return (epoch + 1) / 100.0


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



def epsilon_scheduler(epochs, start_eps=0.1, end_eps=0.03, warmup_epochs=15):
    # Start soft (0.1), decay linearly to sharp (0.03) over the warmup period, then keep flat
    schedule = np.linspace(start_eps, end_eps, warmup_epochs)
    flat = np.full(epochs - warmup_epochs, end_eps)
    return np.concatenate([schedule, flat])


class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps):
        super(WarmUpCosineDecay, self).__init__()
        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Linear Warmup
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        
        # Cosine Decay
        decay_steps = tf.maximum(step - self.warmup_steps, 0.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * decay_steps / (self.total_steps - self.warmup_steps)))
        decayed_lr = self.initial_lr * cosine_decay
        
        # Return Warmup if in warmup phase, else Decay
        return tf.where(step < self.warmup_steps, warmup_lr, decayed_lr)