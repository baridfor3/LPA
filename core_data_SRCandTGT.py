# encoding=utf8
import os
from hyper_and_conf import hyper_fn as train_conf
from data import data_setentceToByte_helper
import tensorflow as tf
import numpy as np
cwd = os.getcwd()
PADDED_IMG = 180
PADDED_TEXT = 120

# PADDED_IMG = 50
# PADDED_TEXT = 10
DIC_SRC = cwd + '/data/_BYTE_LEVEL_vocabulary_13k'
DIC_TGT = cwd + '/data/_BYTE_LEVEL_vocabulary_char'


class DatasetManager():
    def __init__(self,
                 source_data_path,
                 target_data_path,
                 batch_size=32,
                 shuffle=100,
                 num_sample=-1,
                 max_length=50,
                 EOS_ID=1,
                 PAD_ID=0,
                 cross_val=[0.89, 0.1, 0.01],
                 byte_token='@@',
                 word_token=' ',
                 split_token='\n',
                 tfrecord_path=None):
        """Short summary.

        Args:
            source_data_path (type): Description of parameter `source_data_path`.
            target_data_path (type): Description of parameter `target_data_path`.
            num_sample (type): Description of parameter `num_sample`.
            batch_size (type): Description of parameter `batch_size`.
            split_token (type): Description of parameter `split_token`.

        Returns:
            type: Description of returned object.

        """
        self.source_data_path = source_data_path
        self.target_data_path = target_data_path
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.byte_token = byte_token
        self.split_token = split_token
        self.word_token = word_token
        self.EOS_ID = EOS_ID
        self.PAD_ID = PAD_ID
        self.shuffle = shuffle
        self.max_length = max_length
        self.cross_val = cross_val
        self.dict_conver = data_setentceToByte_helper.Subtokenizer(DIC_TGT)
        self.tfrecord_path = tfrecord_path

    def corpus_length_checker(self, data=None, re=False):
        self.short_20 = 0
        self.median_50 = 0
        self.long_100 = 0
        self.super_long = 0
        for k, v in enumerate(data):
            v = v.split(self.word_token)
            v_len = len(v)
            if v_len <= 20:
                self.short_20 += 1
            if v_len > 20 and v_len <= 50:
                self.median_50 += 1
            if v_len > 50 and v_len <= 100:
                self.long_100 += 1
            if v_len > 100:
                self.super_long += 1
        if re:
            print("short: %d" % self.short_20)
            print("median: %d" % self.median_50)
            print("long: %d" % self.long_100)
            print("super long: %d" % self.super_long)

    def decode(self, string):
        return self.dict_conver.decode(string)

    def encode(self, string):
        return self.dict_conver.encode(string, add_eos=False)

    def one_file_encoder(self, file_path, num=None):
        with tf.io.gfile.GFile(file_path, "r") as f:
            raw_data = f.readlines()
            re = []
            if num is None:
                for d in raw_data:
                    re.append(self.encode(d))
            else:
                text = raw_data[num].split(":")[1].lower().rstrip().strip()

                re = self.encode(text)
            f.close()
        return re

    def one_file_decoder(self, file_path, line_num=None):
        with tf.io.gfile.GFile(file_path, "r") as f:
            raw_data = f.readlines()
            re = []
            if line_num is None:
                for d in raw_data:
                    re.append(self.decode(d))
                f.close()
            else:
                re.append(self.decode(raw_data[line_num]))
        return re

    def create_dataset(self, files):
        def _parse_example(serialized_example):
            """Return inputs and targets Tensors from a serialized tf.Example."""
            data_fields = {
                "text": tf.io.VarLenFeature(tf.int64),
                "img": tf.io.VarLenFeature(tf.float32),
                "ratio": tf.io.VarLenFeature(tf.float32)
            }
            # import pdb;pdb.set_trace()
            parsed = tf.io.parse_single_example(serialized=serialized_example,
                                                features=data_fields)
            img = tf.sparse.to_dense(parsed["img"])
            text = tf.sparse.to_dense(parsed["text"])
            ratio = tf.sparse.to_dense(parsed["ratio"])
            return img, text, ratio

        def _filter_max_length(example, max_length=150):
            return tf.logical_and(
                tf.size(input=example[0]) <= PADDED_IMG * 6144,
                # tf.size(example[1]) <= PADDED_TEXT)
                tf.greater_equal(example[2], 0.1)[0])
            # tf.greater_equal(example[2], tf.constant(0.1))[0])

        def reshape_data(src, tgt):
            return tf.reshape(src, [-1, 32, 64, 3]), tgt

        def format_data(img, text, ratio):
            return (
                tf.reshape(img, (-1, 6144))[:PADDED_IMG, :],
                tf.cast(text[:PADDED_TEXT], dtype=tf.int64))

        options = tf.data.Options()
        options.experimental_deterministic = False
        dataset = files.interleave(
            lambda file: tf.data.TFRecordDataset(
                file, compression_type='GZIP', buffer_size=8 * 1000 * 10000),
            cycle_length=tf.data.experimental.AUTOTUNE,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).with_options(
                options)
        dataset = dataset.map(_parse_example,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.filter(lambda x, y, z: _filter_max_length(
            (x, y, z), 100))
        dataset = dataset.map(format_data,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(reshape_data,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def create_text_dataset(self, files):
        def _parse_example(serialized_example):
            """Return inputs and targets Tensors from a serialized tf.Example."""
            data_fields = {
                "text": tf.io.VarLenFeature(tf.int64),
                "img": tf.io.VarLenFeature(tf.float32),
                "ratio": tf.io.VarLenFeature(tf.float32)
            }
            parsed = tf.io.parse_single_example(serialized=serialized_example,
                                                features=data_fields)
            img = tf.sparse.to_dense(parsed["img"])
            text = tf.sparse.to_dense(parsed["text"])
            ratio = tf.sparse.to_dense(parsed["ratio"])
            return img, text, ratio

        def format_data(img, text, ratio):
            return tf.cast(text[:80], dtype=tf.int64)

        options = tf.data.Options()
        options.experimental_deterministic = False
        dataset = files.interleave(
            lambda file: tf.data.TFRecordDataset(
                file, compression_type='GZIP', buffer_size=8 * 1000 * 1000),
            cycle_length=12,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).with_options(
                options)
        dataset = dataset.map(_parse_example,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(format_data)
        return dataset

    def create_external_dataset(self, data_path):
        def _parse_example(serialized_example):
            """Return inputs and targets Tensors from a serialized tf.Example."""
            data_fields = {
                "src": tf.io.VarLenFeature(tf.int64),
                "tgt": tf.io.VarLenFeature(tf.int64)
            }
            parsed = tf.io.parse_single_example(serialized_example,
                                                data_fields)
            src = tf.sparse.to_dense(parsed["src"])
            tgt = tf.sparse.to_dense(parsed["tgt"])
            return src, tgt

        def _filter_max_length(example, max_length=256):
            return example[0][:PADDED_TEXT]

        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(_parse_example,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x, y: _filter_max_length(
            (x, y), self.max_length))
        return dataset

    def get_raw_train_dataset(self):
        files = tf.data.Dataset.list_files(
            [
                self.tfrecord_path +
                "/lip_reading_raw_pre_train/train_TFRecord_*",
                self.tfrecord_path + "/lip_reading_raw_main/train_TFRecord_*",
            ],
            shuffle=True)
        files = files.shuffle(10000)
        return self.create_dataset(files)

    def get_text_train_dataset(self):
        files = tf.data.Dataset.list_files(
            [
                self.tfrecord_path +
                "/lip_reading_raw_pre_train/train_TFRecord_*",
            ],
            shuffle=True)
        files = files.repeat().shuffle(5000)
        return self.create_text_dataset(files)

    def get_external_dataset(self):
        return self.create_external_dataset("./data/train_TFRecord_0")
