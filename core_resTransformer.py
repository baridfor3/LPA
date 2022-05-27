# encoder=utf8
import tensorflow as tf
from hyper_and_conf import hyper_layer
import core_resnet as Res
L2_WEIGHT_DECAY = 1e-6
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


class ResSA(tf.keras.layers.Layer):
    def __init__(self,
                 max_seq_len,
                 vocabulary_size,
                 embedding_size=1024,
                 num_units=1024,
                 num_heads=16,
                 dropout=0.1,
                 eos_id=1,
                 pad_id=0,
                 resNet=None):
        super(ResSA, self).__init__(name="ResSA")
        self.max_seq_len = max_seq_len
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout = dropout
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.resNet = resNet

    def build(self, input_shape):
        self.time_norm = tf.keras.layers.BatchNormalization(
            axis=1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)
        # FFN
        self.FFN = hyper_layer.NormBlock(
            hyper_layer.Feed_Forward_Network(self.num_units, self.dropout), self.dropout)
        # LPA
        self.LPA_init = hyper_layer.LPA(
            self.num_units, self.num_heads, (2, 2), [2, 2], self.dropout)
        self.LPA_256 = hyper_layer.LPA(
            self.num_units, self.num_heads, (4, 4), [4, 4], self.dropout)
        self.LPA_512 = hyper_layer.LPA(
            self.num_units, self.num_heads, (2, 4), [2, 4], self.dropout)
        self.LPA_1024 = hyper_layer.LPA(
            self.num_units, self.num_heads, (2, 2), [2, 2], self.dropout)
        self.LPA_2048 = hyper_layer.LPA(
            self.num_units, self.num_heads, (1, 2), [1, 2], self.dropout)
        # Weighted scale aggregation
        self.offset_stage1 = self.add_weight(name='offset1', shape=[1])
        self.offset_stage2 = self.add_weight(name='offset2', shape=[1])
        self.offset_stage3 = self.add_weight(name='offset3', shape=[1])
        self.offset_stage4 = self.add_weight(name='offset4', shape=[1])
        self.offset_stage5 = self.add_weight(name='offset5', shape=[1])
        self.stage1_LN = hyper_layer.LayerNorm()
        self.stage2_LN = hyper_layer.LayerNorm()
        self.stage3_LN = hyper_layer.LayerNorm()
        self.stage4_LN = hyper_layer.LayerNorm()
        self.stage5_LN = hyper_layer.LayerNorm()
        super(ResSA, self).build(input_shape)

    def call(self, inputs, attention_bias, training):
        # if training:
        #     droput = self.dropout
        # else:
        #     droput = 0
        attention_weights = {}
        x = inputs
        with tf.name_scope("stacked_encoder"):
            length = tf.shape(inputs)[1]
            batch = tf.shape(inputs)[0]
            res_paddings = tf.reduce_sum(tf.reshape(x, [-1, 32 * 64 * 3]), -1)
            res_paddings = tf.cast(tf.not_equal(res_paddings, 0.), tf.float32)
            padding_bias = tf.reshape(res_paddings, [batch, length, 1])
            x = tf.reshape(x, [-1, 32, 64, 3])
            x = tf.image.per_image_standardization(x)

            # stage_1
            x = Res.ResNet50_1(x)
            self_att, att_w = self.LPA_init(tf.reshape(
                x, [-1, length, 8, 16, 64]), attention_bias, training)
            attention_weights[
                'stage_1'] = att_w
            att = self.stage1_LN(self_att) * self.offset_stage1

            # stage=2
            x = Res.ResNet50_2(x)

            self_att, att_w = self.LPA_256(tf.reshape(
                x, [-1, length, 8, 16, 256]), attention_bias, training)
            attention_weights[
                'stage_2'] = att_w
            att = att + self.stage2_LN(self_att) * self.offset_stage2

            # stage = 3
            x = Res.ResNet50_3(x)
            self_att, att_w = self.LPA_512(tf.reshape(
                x, [-1, length, 4, 8, 512]), attention_bias, training)
            attention_weights[
                'stage_3'] = att_w
            att = att + self.stage3_LN(self_att) * self.offset_stage3

            # stage=4
            x = Res.ResNet50_4(x)
            self_att, att_w = self.LPA_1024(tf.reshape(
                x, [-1, length, 2, 4, 1024]), attention_bias, training)
            attention_weights[
                'stage_4'] = att_w
            att = att + self.stage4_LN(self_att) * self.offset_stage4

            # stage=5
            x = Res.ResNet50_5(x)
            self_att, att_w = self.LPA_2048(tf.reshape(
                x, [-1, length, 1, 2, 2048]), attention_bias, training)
            attention_weights[
                'stage_5'] = att_w
            att = att + self.stage5_LN(self_att) * self.offset_stage5
            ##
            # final
            # padding_bias = tf.reshape(res_paddings, [-1, length, 1])
            x = self.FFN(att * padding_bias)
            return x * padding_bias

    def get_resNet(self, training):
        if training:
            res = tf.keras.models.load_model('pre_train/res50_pre_all')
        else:
            res = tf.keras.applications.resnet50.ResNet50(
                include_top=False, weights=None, input_shape=[32, 64, 3])
        return res

    def get_config(self):
        c = {
            "max_seq_len": self.max_seq_len,
            "vocabulary_size": self.vocabulary_size,
            "embedding_size": self.embedding_size,
            "num_units": self.num_units,
            "num_heads": self.num_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "dropout": self.dropout,
        }
        return c


class LinearDecoder(tf.keras.layers.Layer):
    def __init__(
            self,
            max_seq_len,
            vocabulary_size,
            embedding_size=512,
            num_units=512,
            num_heads=6,
            num_decoder_layers=6,
            dropout=0.4,
            eos_id=1,
            pad_id=0,
    ):
        super(LinearDecoder, self).__init__(name="Linear_decoder")
        self.max_seq_len = max_seq_len
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.attention_weights = {}

    def build(self, input_shape):

        self.decoder_output = hyper_layer.LayerNorm()
        self.stacked_decoder = []
        # self.ln = hyper_layer.LayerNorm()
        for i in range(self.num_decoder_layers):
            self_attention = hyper_layer.SelfAttention(
                num_heads=self.num_heads,
                num_units=self.num_units,
                dropout=self.dropout,
            )

            attention = hyper_layer.Attention(num_heads=self.num_heads,
                                              num_units=self.num_units,
                                              dropout=self.dropout)
            ffn = hyper_layer.Feed_Forward_Network(num_units=4 *
                                                   self.num_units,
                                                   dropout=self.dropout)
            offset_inputs = self.add_weight(name='offset', shape=[1])
            offset_org = self.add_weight(name='offset', shape=[1])
            self.stacked_decoder.append([
                hyper_layer.NormBlock(self_attention, self.dropout),
                hyper_layer.NormBlock(attention, self.dropout),
                hyper_layer.NormBlock(ffn, self.dropout), offset_inputs,
                offset_org
            ])
        # self.mask = tf.keras.layers.Masking(0)
        super(LinearDecoder, self).build(input_shape)

    def call(self,
             inputs,
             enc,
             decoder_self_attention_bias,
             attention_bias,
             cache=None,
             training=False,
             GPT=False):
        org = inputs
        with tf.name_scope("stacked_decoder"):
            for index, layer in enumerate(self.stacked_decoder):
                self_att = layer[0]
                att = layer[1]
                ffn = layer[2]
                offset_inputs = layer[3]
                offset_org = layer[4]
                layer_name = "layer_%d" % index
                layer_cache = cache[layer_name] if cache is not None else None
                with tf.name_scope("layer_%d" % index):
                    inputs = self_att(
                        inputs,
                        decoder_self_attention_bias,
                        training=training,
                        cache=layer_cache,
                    )
                    attn_weights_block1 = self_att.layer.get_attention_weights(
                    )
                    if GPT is not True:
                        inputs = att(inputs,
                                     enc,
                                     attention_bias,
                                     training=training)
                    attn_weights_block2 = att.layer.get_attention_weights()
                    self.attention_weights['decoder_layer{}_block1'.format(
                        index + 1)] = attn_weights_block1
                    self.attention_weights['decoder_layer{}_block2'.format(
                        index + 1)] = attn_weights_block2
                    inputs = ffn(inputs, training=training)
                    inputs = offset_inputs * inputs + offset_org * org
                    org = inputs
        return self.decoder_output(inputs)

    def get_attention_weights(self):
        return self.attention_weights

    def get_config(self):
        c = {
            "max_seq_len": self.max_seq_len,
            "vocabulary_size": self.vocabulary_size,
            "embedding_size": self.embedding_size,
            "num_units": self.num_units,
            "num_heads": self.num_heads,
            "num_decoder_layers": self.num_decoder_layers,
            "dropout": self.dropout,
        }
        return c
