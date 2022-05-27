# encoding=utf8
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
import hyper_and_conf.conf_metrics as conf_metrics
import hyper_and_conf.hyper_fn as hyper_fn
import numpy as np
from tensorflow.python.keras.losses import Loss
from tensorflow.python.ops import math_ops


class Mean_MetricLayer(tf.keras.layers.Layer):

    def __init__(self, name="custom_mean"):
        # self.mean_fn = tf.keras.metrics.Mean(name)
        self.metric_name = name
        super(Mean_MetricLayer, self).__init__(name=name)

    def build(self, input_shape):
        """"Builds metric layer."""
        super(Mean_MetricLayer, self).build(input_shape)

    def call(self, inputs, penalty=1, ):
        self.add_loss(inputs * penalty)
        self.add_metric(inputs, name=self.metric_name, aggregation="mean")
        return inputs


class Test_Unigram_BLEU_Metric(tf.keras.metrics.Metric):
    def __init__(self):
        super(Test_Unigram_BLEU_Metric, self).__init__()
        self.total = self.add_weight(name='UnigramBLEU', initializer='zeros')
        # self.values = self.add_weight(name='BLEU', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        value, _ = conf_metrics.approx_unigram_bleu(y_true, y_pred)
        # if sample_weight is not None:
        #     sample_weight = tf.cast(sample_weight, 'float32')
        #     values = tf.multiply(values, sample_weight)
        self.count.assign_add(1)
        self.total.assign_add(value)

    def result(self):

        return math_ops.div_no_nan(self.total, self.count)
        # return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total.assign(0.)
        self.count.assign(0)


class Loss_Metric(tf.keras.metrics.Metric):
    def __init__(self):
        super(Loss_Metric, self).__init__()
        self.total = self.add_weight(name='loss', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')
        # self.total = 0
        # self.count = 0

    def update_state(self, value, sample_weight=None):
        value = tf.py_function(lambda x: tf.cast(x, tf.float32), [value],
                               tf.float32)
        # value, _ = conf_metrics.wer_score(y_true, y_pred)
        # if sample_weight is not None:
        #     sample_weight = tf.cast(sample_weight, 'float32')
        #     values = tf.multiply(values, sample_weight)
        # self.count = tf.add(self.count, 1)
        # self.total = tf.add(self.total, value)
        self.count += 1
        self.total += value

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total = 0
        self.count = 0


class Test_Wer_Metric(tf.keras.metrics.Metric):
    def __init__(self):
        super(Test_Wer_Metric, self).__init__()
        self.total = self.add_weight(name='WER', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        value, _ = conf_metrics.wer_score(y_true, y_pred)
        self.count += 1
        self.total += value

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total = 0
        self.count = 0


class Onehot_CrossEntropy(Loss):
    def __init__(self, vocab_size, mask_id=0, smoothing=0.0):

        super(Onehot_CrossEntropy,
              self).__init__(reduction=tf.keras.losses.Reduction.NONE,
                             name="Onehot_CrossEntropy")
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.smoothing = smoothing

    def call(self, true, pred):
        batch_size = tf.shape(true)[0]
        true = tf.reshape(true, [batch_size, -1])
        loss = conf_metrics.onehot_loss_function(true=true,
                                                 pred=pred,
                                                 mask_id=self.mask_id,
                                                 smoothing=self.smoothing,
                                                 vocab_size=self.vocab_size)
        return loss


class FirstToken_CrossEntropy_layer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, label_smoothing):
        super(FirstToken_CrossEntropy_layer, self).__init__()
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.log = tf.keras.metrics.Mean("first_loss")

    def build(self, input_shape):
        super(FirstToken_CrossEntropy_layer, self).build(input_shape)

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "label_smoothing": self.label_smoothing,
        }

    def call(self, inputs):
        targets, logits = inputs[0], inputs[1]
        loss = conf_metrics.firstToken_loss(targets,
                                            logits,
                                            smoothing=self.label_smoothing,
                                            vocab_size=self.vocab_size)
        self.add_loss(0.5 * loss)
        m = self.log(loss)
        self.add_metric(m)
        return logits


class CrossEntropy_layer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, label_smoothing):
        super(CrossEntropy_layer, self).__init__()
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing

    def build(self, input_shape):
        super(CrossEntropy_layer, self).build(input_shape)

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "label_smoothing": self.label_smoothing,
        }

    def call(self, inputs):
        targets, logits = inputs[0], inputs[1]
        loss = conf_metrics.onehot_loss_function(
            targets,
            logits,
            smoothing=self.label_smoothing,
            vocab_size=self.vocab_size)
        self.add_loss(loss)
        return logits


class CTC_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(CTC_layer, self).__init__()

    def build(self, input_shape):
        super(CTC_layer, self).build(input_shape)

    def call(self, inputs):
        targets, logits = inputs[0], inputs[1]
        loss = tf.nn.ctc_loss(
            targets, logits, tf.fill([tf.shape(targets)[0]], tf.shape(targets)[1]), tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1]), logits_time_major=False, blank_index=-1
        )
        loss = tf.reduce_mean(loss)
        self.add_loss(loss)
        return logits


class MetricLayer(tf.keras.layers.Layer):
    """Custom a layer of metrics for Transformer model."""

    def __init__(self, vocab_size):
        super(MetricLayer, self).__init__()
        self.vocab_size = vocab_size
        self.metric_mean_fns = []

    def build(self, input_shape):
        """"Builds metric layer."""
        self.metric_mean_fns = [
            (tf.keras.metrics.Mean("approx_4-gram_bleu"),
             conf_metrics.approx_bleu),
            (tf.keras.metrics.Mean("approx_unigram_bleu"),
             conf_metrics.approx_unigram_bleu),
            (tf.keras.metrics.Mean("wer"), conf_metrics.wer_score),
            (tf.keras.metrics.Mean("accuracy"), conf_metrics.padded_accuracy),
            (tf.keras.metrics.Mean("accuracy_top5"),
             conf_metrics.padded_accuracy_top5),
            (tf.keras.metrics.Mean("accuracy_per_sequence"),
             conf_metrics.padded_sequence_accuracy),
        ]
        super(MetricLayer, self).build(input_shape)

    def get_config(self):
        return {"vocab_size": self.vocab_size}

    def call(self, inputs):
        targets, logits = inputs[0], inputs[1]
        for mean, fn in self.metric_mean_fns:
            m = mean(*fn(targets, logits))
            self.add_metric(m)
        # else:
        #     for mean, fn in self.metric_mean_fns:
        #         m = mean(*fn(logits, targets))
        #         self.add_metric(m)
        return logits


class Dynamic_LearningRate(Callback):
    def __init__(self,
                 init_lr,
                 num_units,
                 learning_rate_warmup_steps,
                 verbose=0):
        super(Dynamic_LearningRate, self).__init__()
        self.init_lr = init_lr
        self.num_units = num_units
        self.learning_rate_warmup_steps = learning_rate_warmup_steps
        self.verbose = verbose
        self.sess = tf.compat.v1.keras.backend.get_session()
        self._total_batches_seen = 0
        self.current_lr = 0

    def on_train_begin(self, logs=None):
        self.current_lr = hyper_fn.get_learning_rate(
            self.init_lr, self.num_units, self._total_batches_seen,
            self.learning_rate_warmup_steps)
        lr = float(self.current_lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nStart  learning ' 'rate from %s.' % (lr))

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            self.current_lr = hyper_fn.get_learning_rate(
                self.init_lr, self.num_units, self._total_batches_seen,
                self.learning_rate_warmup_steps)
        except Exception:  # Support for old API for backward compatibility
            self.current_lr = self.init_lr
        lr = float(self.current_lr)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            tf.compat.v1.logging.info('\nEpoch %05d: Changing  learning '
                                      'rate to %s.' % (batch + 1, lr))

    def on_batch_end(self, batch, logs=None):
        self._total_batches_seen += 1
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
