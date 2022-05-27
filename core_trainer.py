# -*- coding: utf-8 -*-
# code warrior: Barid
"""
    This is a naive seq2seq model wrapping commone functions.
"""
import tensorflow as tf
from hyper_and_conf import hyper_train


class NaiveSeq2Seq(tf.keras.Model):
    def __init__(self, vocabulary_size, **kwargs):
        super(NaiveSeq2Seq, self).__init__(name="NaiveSeq2Seq")
        try:
            sos = kwargs["sos_id"]
        except Exception:
            sos = 0
        try:
            eos = kwargs["eos_id"]
        except Exception:
            eos = 0
        try:
            self.label_smoothing = kwargs["label_smoothing"]

        except Exception:
            self.label_smoothing = 0.1
        # try:
        #     self.gradient_tower = kwargs["gradient_tower"]
        # except Exception:
        #     self.gradient_tower = None
        self.eos = eos
        self.sos = sos
        self.vocabulary_size = vocabulary_size

        # def build(self, input_shape):
        self.total_loss = hyper_train.Mean_MetricLayer("loss")
        self.grad_norm_ratio = hyper_train.Mean_MetricLayer("grad_norm_ratio")
        self.tokenPerS = hyper_train.Mean_MetricLayer("tokens/batch")
        self.finetune = False
        self.seq2seq_loss_FN = hyper_train.CrossEntropy_layer(self.vocabulary_size,
                                                              self.label_smoothing)
        self.ctc_loss_FN = hyper_train.CTC_layer()

        self.seq2seq_metric = hyper_train.MetricLayer(self.vocabulary_size)

    def compile(self, optimizer):
        super(NaiveSeq2Seq, self).compile()
        self.optimizer = optimizer

    def call(self, inputs, training):
        assert True, "The call function should be reimplemented."
        return None

    def seq2seq_training(self, x, de_real_y, y, mode='cross', **kwargs):
        with tf.GradientTape(persistent=False) as model_tape:
            mt_logits = self.call((x, de_real_y), training=True, **kwargs)
            if mode == "cross":
                loss = self.seq2seq_loss_FN([y, mt_logits], auto_loss=False)
            else:
                loss = self.ctc_loss_FN([y, mt_logits], auto_loss=False)
                mt_logits = tf.transpose(mt_logits, [1, 0, 2])
            model_gradients = model_tape.gradient(loss,
                                                  self.trainable_variables)
            model_gradients, grad_norm = tf.clip_by_global_norm(
                model_gradients, 1.0)
            self.optimizer.apply_gradients(
                zip(model_gradients, self.trainable_variables))
            self.grad_norm_ratio(grad_norm)
            self.total_loss(loss)
            self.seq2seq_metric([y, mt_logits])
            batch_size = tf.shape(x)[0]
            self.tokenPerS(
                tf.cast(
                    tf.math.multiply(batch_size,
                                     (tf.shape(x)[1] + tf.shape(y)[1])),
                    tf.float32))
            return loss
