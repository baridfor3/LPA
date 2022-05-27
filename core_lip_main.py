# encoding=utf8
import tensorflow as tf
from core_resTransformer import LinearDecoder, ResSA
from hyper_and_conf import hyper_layer
from hyper_and_conf import hyper_beam_search as beam_search
import json
from hyper_and_conf import hyper_fn,  hyper_util, hyper_train


L2_WEIGHT_DECAY = 1e-6
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


PADDED_IMG = 180
PADDED_TEXT = 120

FEATURE_IMG = 32 * 64 * 3


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


class Daedalus(NaiveSeq2Seq):
    def __init__(self,
                 max_seq_len=50,
                 vocabulary_size=55,
                 embedding_size=1024,
                 num_units=1024,
                 num_heads=16,
                 num_decoder_layers=6,
                 dropout=0.1,
                 eos_id=1,
                 pad_id=0,
                 mask_id=2,
                 resNet=None,
                 mode='LIP',
                 ** kwargs):
        super(Daedalus, self).__init__(vocabulary_size, name="lip_reading")
        self.max_seq_len = max_seq_len
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.EOS_ID = eos_id
        self.PAD_ID = pad_id
        self.MASK_ID = mask_id
        self.resNet = resNet
        self.embedding_block = hyper_layer.EmbeddingSharedWeights(
            vocab_size=self.vocabulary_size,
            num_units=self.num_units,
            pad_id=0,
            name="word_embedding",
        )

        self.res_encoder = ResSA(
            self.max_seq_len,
            self.vocabulary_size,
            self.embedding_size,
            self.num_units,
            self.num_heads,
            self.dropout,
            self.EOS_ID,
            self.PAD_ID,
            resNet=self.resNet,
        )
        self.stacked_decoder = LinearDecoder(
            self.max_seq_len,
            self.vocabulary_size,
            self.embedding_size,
            self.num_units,
            self.num_heads,
            self.num_decoder_layers,
            self.dropout,
            self.EOS_ID,
            self.PAD_ID,
        )
        self.VIS_generator = tf.keras.layers.Dense(self.vocabulary_size)
        self.mode = mode

    def call(self, inputs, training):
        if len(inputs) == 2:
            img_input, tgt_input = inputs[0], inputs[1]
        else:
            img_input, tgt_input = inputs[0], None
        with tf.name_scope("lip_reading"):
            import pdb
            pdb.set_trace()
            length = tf.shape(input=img_input)[1]
            encoder_outputs = img_input
            img_input = tf.reshape(encoder_outputs, [-1, length, FEATURE_IMG])
            attention_bias = hyper_util.get_padding_bias(
                tf.reduce_sum(img_input, -1))
            if training:
                encoder_outputs = tf.nn.dropout(encoder_outputs,
                                                rate=self.dropout)
            encoder_outputs = self.res_encoder(encoder_outputs,
                                               attention_bias,
                                               training=training)
            # for visual module pre-training
            if self.mode == "VIS":
                return self.VIS_generator(encoder_outputs)

            length = tf.shape(encoder_outputs)[1]
            if tgt_input is None and not training:
                return self.inference(encoder_outputs,
                                      attention_bias,
                                      training=False)
            else:
                logits = self.decode(tgt_input,
                                     encoder_outputs,
                                     attention_bias,
                                     training,
                                     mode=self.mode)
                return logits

    def get_stage_weights(self):
        weights = self.res_encoder.get_attention_weights()
        weight = {}
        for k, v in weights.items():
            weight[k] = v.numpy().tolist()
        with open('./stage_trained.json', 'w') as fp:
            json.dump(weight, fp)
        return weights

    def get_decoder_weights(self):
        weights = self.stacked_decoder.get_attention_weights()
        weight = {}
        for k, v in weights.items():
            weight[k] = v.numpy().tolist()
        with open('./decoder.json', 'w') as fp:
            json.dump(weight, fp)
        return weight

    def decode(self,
               targets,
               encoder_outputs,
               attention_bias,
               training,
               mode='LIP'):
        with tf.name_scope("decode"):
            decoder_inputs = targets
            decoder_inputs = tf.pad(decoder_inputs, [[0, 0], [1, 0]])[:, :-1]

            decoder_inputs = self.embedding_block(decoder_inputs)
            length = tf.shape(decoder_inputs)[1]
            pos_encoding = hyper_fn.get_position_encoding(
                length, self.num_units)
            decoder_inputs = decoder_inputs + pos_encoding
            if mode == "GPT":
                # for decoder pre-training
                decoder_self_attention_bias = self.window_bias(length)
                outputs = self.stacked_decoder(
                    decoder_inputs,
                    encoder_outputs,
                    decoder_self_attention_bias,
                    attention_bias,
                    training=training,
                    GPT=True,
                )
            else:
                import pdb
                pdb.set_trace()
                decoder_self_attention_bias = hyper_util.get_decoder_self_attention_bias(
                    length)
                outputs = self.stacked_decoder(
                    decoder_inputs,
                    encoder_outputs,
                    decoder_self_attention_bias,
                    attention_bias,
                    training=training,
                    GPT=False,
                )
            logits = self.embedding_block(outputs, linear=True)

            return logits

    def train_step(self, data):
        ((x, y), ) = data
        # mask_x, mask_y = hyper_util.pad_tensors_to_same_length(mask_x, mask_y)
        de_real_y = tf.pad(y, [[0, 0], [1, 0]],
                           constant_values=self.TGT_SOS_ID)[:, :-1]
        if self.mode == 'VIS':
            _ = self.seq2seq_training(x, de_real_y, y, mode='CTC')
        return {m.name: m.result() for m in self.metrics}

    def pre_processing(self, src, tgt):
        attention_bias = hyper_util.get_padding_bias(src)
        decoder_padding = hyper_util.get_padding_bias(tgt)
        decoder_self_attention_bias = hyper_util.get_decoder_self_attention_bias(
            tf.shape(tgt)[1])
        return attention_bias, decoder_self_attention_bias, decoder_padding

    def _get_symbols_to_logits_fn(self, max_decode_length, training):
        timing_signal = hyper_fn.get_position_encoding(max_decode_length + 1,
                                                       self.num_units)
        decoder_self_attention_bias = hyper_util.get_decoder_self_attention_bias(
            max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            decoder_input = ids[:, -1:]
            decoder_input = self.embedding_block(decoder_input)
            decoder_input += timing_signal[i:i + 1]
            self_attention_bias = decoder_self_attention_bias[:, :, i:i +
                                                              1, :i + 1]
            decoder_outputs = self.stacked_decoder(
                decoder_input,
                cache.get("encoder_outputs"),
                self_attention_bias,
                cache.get("encoder_decoder_attention_bias"),
                training=training,
                cache=cache,
            )
            logits = self.embedding_block(decoder_outputs, linear=True)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return symbols_to_logits_fn

    def inference(self, encoder_outputs, encoder_decoder_attention_bias,
                  training):
        """Return predicted sequence."""
        batch_size = tf.shape(encoder_outputs)[0]
        # input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = self.max_seq_len

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(
            max_decode_length, training)
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)
        init_decode_length = 0
        dim_per_head = self.num_units // self.num_heads
        cache = {
            "layer_%d" % layer: {
                "k":
                tf.zeros([
                    batch_size, self.num_heads, init_decode_length,
                    dim_per_head
                ]),
                "v":
                tf.zeros([
                    batch_size, self.num_heads, init_decode_length,
                    dim_per_head
                ]),
            }
            for layer in range(self.num_decoder_layers)
        }

        cache["encoder_outputs"] = encoder_outputs
        cache[
            "encoder_decoder_attention_bias"] = encoder_decoder_attention_bias
        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.vocabulary_size,
            beam_size=64,
            alpha=1,
            max_decode_length=max_decode_length,
            eos_id=self.EOS_ID,
        )
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {"outputs": top_decoded_ids, "scores": top_scores}

    def padding_bias(self, padding, eye=False):
        if eye:
            length = tf.shape(padding)[1]
            self_ignore = tf.eye(length, dtype=tf.float32)
            self_ignore = tf.expand_dims(tf.expand_dims(self_ignore, axis=0),
                                         axis=0)
            padding = tf.expand_dims(tf.expand_dims(padding, axis=1), axis=1)
            padding = tf.cast(tf.cast((self_ignore + padding), tf.bool),
                              tf.float32)
        else:
            padding = tf.expand_dims(tf.expand_dims(padding, axis=1), axis=1)

        attention_bias = padding * (-1e9)
        return attention_bias

    def get_config(self):
        c = {
            "max_seq_len": self.max_seq_len,
            "vocabulary_size": self.vocabulary_size,
            "embedding_size": self.embedding_size,
            "num_units": self.num_units,
            "num_heads": self.num_heads,
            "num_decoder_layers": self.num_decoder_layers,
            "num_encoder_layers": self.num_encoder_layers,
            "dropout": self.dropout,
        }
        return c

    def padding(self, padding):
        padding = tf.expand_dims(padding, axis=-1)
        return padding

    def window_bias(self, length):
        with tf.name_scope("decoder_self_attention_bias"):
            valid_locs = tf.linalg.band_part(tf.ones([length, length]), 3,
                                             3)
            valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
            decoder_bias = -1e9 * (1.0 - valid_locs)
        return decoder_bias

    def get_cache(self):
        return self.cache
