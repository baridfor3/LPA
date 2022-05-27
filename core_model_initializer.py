# encoding=utf-8
import sys
from hyper_and_conf import hyper_param as hyperParam
from hyper_and_conf import hyper_train, hyper_optimizer
import core_lip_main
import core_data_SRCandTGT
import tensorflow as tf
import numpy as np
L2_WEIGHT_DECAY = 1e-6
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]
src_data_path = [DATA_PATH + "/corpus/lip_corpus.txt"]

tgt_data_path = [DATA_PATH + "/corpus/lip_corpus.txt"]

TFRECORD = DATA_PATH + "/lip_reading_data"

PADDED_IMG = 180
PADDED_TEXT = 120


def get_vgg(self):
    if tf.io.gfile.exists('pre_train/vgg16_pre_all'):
        vgg16 = tf.keras.models.load_model('pre_train/vgg16_pre_all')
    else:
        vgg16 = tf.keras.applications.vgg16.VGG16(include_top=True,
                                                  weights='imagenet')
    return vgg16


TRAIN_MODE = 'large'
hp = hyperParam.HyperParam(TRAIN_MODE)
PAD_ID_int64 = tf.cast(hp.PAD_ID, tf.int64)
PAD_ID_float32 = tf.cast(hp.PAD_ID, tf.float32)

data_manager = core_data_SRCandTGT.DatasetManager(
    src_data_path,
    tgt_data_path,
    batch_size=hp.batch_size,
    PAD_ID=hp.PAD_ID,
    EOS_ID=hp.EOS_ID,
    shuffle=hp.data_shuffle,
    max_length=hp.max_sequence_length,
    tfrecord_path=TFRECORD)


def get_hp():
    return hp


def input_fn(flag="TRAIN"):
    if flag == "VAL":
        dataset = data_manager.get_raw_val_dataset()
    else:
        if flag == "TEST":
            dataset = data_manager.get_raw_test_dataset()
        else:
            if flag == "TRAIN":
                dataset = data_manager.get_raw_train_dataset()
            else:
                assert ("data error")
    return dataset


def map_data_for_feed_pertunated(x, y):
    return ((x, randomly_pertunate_input(y)), y)


def map_data_for_feed(x, y):
    return ((x, y), y)


def map_data_for_text(x):
    return ((x, x), x)


def randomly_pertunate_input(x):
    determinater = np.random.randint(10)
    if determinater > 3:
        return x
    else:
        index = np.random.randint(2, size=(1, 80))
        x = x * index
    return x


def pad_sample(dataset, batch_size):
    dataset = dataset.padded_batch(
        batch_size,
        (
            [PADDED_IMG, 32, 64, 3],  # source vectors of unknown size
            [PADDED_TEXT]),  # target vectors of unknown size
        drop_remainder=True)

    return dataset


def pad_text_sample(dataset, batch_size):
    dataset = dataset.padded_batch(
        hp.batch_size,
        [120],  # target vectors of unknown size
        drop_remainder=True)

    return dataset


def reshape_data(src, tgt):
    return tf.reshape(src, [-1, 32, 64, 3]) / 127.5 - 1.0, tgt


def map_data_for_val(src, tgt):
    return src, tgt


def train_Transformer_input():
    dataset = data_manager.get_text_train_dataset()
    dataset = pad_text_sample(dataset, batch_size=hp.batch_size)

    dataset = dataset.map(map_data_for_text)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def train_input(seq2seq=True, pertunate=False):

    dataset = input_fn('TRAIN')
    dataset = pad_sample(dataset, batch_size=hp.batch_size)
    dataset = dataset.map(map_data_for_feed)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def get_image_processor():
    # with tf.device("/cpu:0"):
    if tf.io.gfile.exists('pre_train/res50_pre_all'):
        res = tf.keras.models.load_model('pre_train/res50_pre_all')
    else:
        res = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights=None,
                                                      input_shape=[32, 64, 3])
        # pooling='avg',
        # classes=10000)
        res.save('pre_train/res50_pre_all')
    return res


def trainer(mode):
    daedalus = core_lip_main.Daedalus(hp.max_sequence_length,
                                      hp.vocabulary_size,
                                      hp.embedding_size,
                                      hp.num_units,
                                      hp.num_heads,
                                      hp.num_decoder_layers,
                                      hp.dropout,
                                      hp.EOS_ID,
                                      hp.PAD_ID,
                                      hp.MASK_ID,
                                      resNet=None, mode=mode)
    return daedalus


def get_resNet(training):
    if training:
        res = tf.keras.models.load_model('pre_train/res50_pre_all')
        # res.trainable=False
    else:
        res = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights=None,
                                                      input_shape=[32, 64, 3])
    return res


def get_optimizer():
    return tf.keras.optimizers.Adam(
        beta_1=0.1,
        beta_2=0.98,
        epsilon=1.0e-9,
        clipnorm=True,
    )


def get_callbacks():
    lr_fn = hyper_optimizer.LearningRateFn(hp.lr, hp.num_units,
                                           hp.learning_warmup)
    LRschedule = hyper_optimizer.LearningRateScheduler(lr_fn, 0)
    TFboard = tf.keras.callbacks.TensorBoard(log_dir=hp.model_summary_dir,
                                             write_grads=True,
                                             histogram_freq=1000,
                                             embeddings_freq=1000,
                                             write_images=True,
                                             update_freq=100)
    TFchechpoint = tf.keras.callbacks.ModelCheckpoint(
        hp.model_checkpoint_dir + '/model.{epoch:02d}.ckpt',
        save_weights_only=True,
        save_freq=100000,
        verbose=1)
    NaNchecker = tf.keras.callbacks.TerminateOnNaN()
    ForceLrReduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy',
                                                         factor=0.2,
                                                         patience=1,
                                                         mode='max',
                                                         min_lr=0.00001)
    GradientBoard = hyper_train.GradientBoard(hp.model_summary_dir)
    return [
        LRschedule,
        TFboard,
        TFchechpoint,
        NaNchecker,
        ForceLrReduce,
    ]
