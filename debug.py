# encode=utf8
import pdb
import core_model_initializer as init
import tensorflow as tf
import core_lip_main
import sys
import os
import json
import numpy as np
cwd = os.getcwd()
sys.path.insert(0, cwd + "/corpus")
sys.path.insert(1, cwd)
DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]

with tf.device("/cpu:0"):

    hp = init.get_hp()
    train_model = core_lip_main.Daedalus(hp.max_sequence_length,
                                         hp.vocabulary_size,
                                         hp.embedding_size,
                                         hp.num_units,
                                         hp.num_heads,
                                         hp.num_decoder_layers,
                                         hp.dropout,
                                         hp.EOS_ID,
                                         hp.PAD_ID,
                                         hp.MASK_ID,
                                         mode='LIP')

    data_manager = init.data_manager

    with open('sample.json') as f:
        x = json.load(f)

    def save_sample(x, y):
        weight = {}
        weight['sample'] = x
        weight['ids'] = y
        with open('./sample.json', 'w') as fp:
            json.dump(weight, fp)
        return weight

    import pdb
    pdb.set_trace()
    ids = train_model((np.array(x['sample']), [[1, 2, 3, 4]]))
    train_model.summary()
    print("#####################")
