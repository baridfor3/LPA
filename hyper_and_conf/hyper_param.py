# encoding=utf-8
import os
cwd = os.getcwd()


class HyperParam:
    def __init__(self,
                 mode,
                 vocab=55,
                 UNK_ID=0,
                 SOS_ID=0,
                 EOS_ID=1,
                 PAD_ID=0,
                 MASK_ID=2):
        self.UNK_ID = UNK_ID
        self.SOS_ID = SOS_ID
        self.EOS_ID = EOS_ID
        self.PAD_ID = PAD_ID
        self.MASK_ID = MASK_ID
        self.model_summary_dir = cwd + "/model_summary"
        self.model_weights_dir = cwd + "/model_weights"
        self.model_checkpoint_dir = cwd + "/model_checkpoint"
        try:
            os.makedirs(self.model_weights_dir)
        except OSError:
            pass
        try:
            os.makedirs(self.model_checkpoint_dir)
        except OSError:
            pass
        try:
            os.makedirs(self.model_summary_dir)
        except OSError:
            pass

        self.vocabulary_size = vocab

        if mode == 'test':
            self.test()
        if mode == 'small':
            self.small()
        if mode == 'large':
            self.large()

    def test(self,
             embedding_size=64 * 8,
             batch_size=8,
             epoch_num=5,
             num_units=64 * 8,
             num_heads=8,
             num_decoder_layers=2,
             max_sequence_length=150,
             epoch=1,
             lr=2,
             clipping=5,
             inference_length=150,
             data_shuffle=1,
             learning_warmup=12000,
             dropout=0.4):

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.lr = lr
        self.clipping = clipping
        self.data_shuffle = data_shuffle
        self.inference_length = inference_length
        self.learning_warmup = learning_warmup

    def small(self,
              embedding_size=128,
              batch_size=8,
              epoch_num=5,
              num_units=128,
              num_heads=4,
              num_decoder_layers=2,
              max_sequence_length=150,
              epoch=1,
              lr=2,
              clipping=5,
              inference_length=150,
              data_shuffle=100,
              dropout=0.1,
              learning_warmup=10000):

        self.embedding_size = embedding_size
        self.batch_size = batch_size * 2
        self.epoch_num = epoch_num
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.lr = lr
        self.clipping = clipping
        self.data_shuffle = data_shuffle
        self.inference_length = inference_length
        self.learning_warmup = learning_warmup

    def large(self,
              embedding_size=64 * 16,
              batch_size=12,
              epoch_num=200,
              num_units=64 * 16,
              num_heads=16,
              num_decoder_layers=6,
              max_sequence_length=100,
              epoch=100,
              lr=0.001,
              clipping=5,
              inference_length=100,
              data_shuffle=45000,
              dropout=0.1,
              learning_warmup=3000):

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.lr = lr
        self.clipping = clipping
        self.data_shuffle = data_shuffle
        self.inference_length = inference_length
        self.learning_warmup = learning_warmup
