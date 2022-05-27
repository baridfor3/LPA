# encoding=utf8
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import data_image_helper
import core_data_SRCandTGT
import six
import visualization
import data_file_helper
cwd = os.getcwd()


image_parser = data_image_helper.data_image_helper()
text_parser = core_data_SRCandTGT.DatasetManager(
    [],
    [],
)

BUFFER_SIZE = 200


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if six.PY3 and isinstance(value, six.text_type):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_file():
    files = []
    with open(TRAIN_PATH, 'r') as f:
        files = f.readlines()
    files = [ROOT_PATH + '/' + f.rstrip() for f in files]
    print(files[:5])
    return files


def sentence_reader(path):
    files = get_file(path)
    for f in files:
        v = f + '.mp4'
        v_data = image_parser.get_raw_dataset(path=v)
        w = f + '.txt'
        w = text_parser.one_file_encoder(w, 0)


def tfrecord_generater(record_dir, raw_data, index, mode='LRS'):
    num_train = 0
    # num_test = 0
    prefix_train = record_dir + "/lrs_TFRecord_"

    def all_exist(filepaths):
        """Returns true if all files in the list exist."""
        for fname in filepaths:
            if not tf.io.gfile.exists(fname):
                return False
        return True

    def txt_line_iterator(path):
        with tf.io.gfile.GFile(path) as f:
            for line in f:
                yield line.strip()

    def dict_to_example(img, txt, cnt):
        """Converts a dictionary of string->int to a tf.Example."""
        features = {}
        features['img'] = _float_feature(img)
        features['text'] = _int64_feature(txt)
        features['ratio'] = _float_feature(cnt)
        return tf.train.Example(features=tf.train.Features(feature=features))

    checker = -1
    shard = 0
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    if mode != 'LRS':
        raw_data, _, word = data_file_helper.read_file(TRAIN_PATH)
        for k, f in enumerate(raw_data):
            v = f
            v_data, cnt = image_parser.get_raw_dataset(path=v)
            if len(v_data) > 0:
                w = word[k]
                w = text_parser.encode(w)
                if len(v_data.shape) == 4:
                    v_data = tf.reshape(v_data, [-1])
                    if checker == shard:
                        pass
                    else:
                        shard = k // BUFFER_SIZE
                        train_writers = tf.io.TFRecordWriter(
                            prefix_train + str(index * 1000000 + shard),
                            options=options)
                    example = dict_to_example(v_data.numpy().tolist(), w, cnt)
                    train_writers.write(example.SerializeToString())
                    checker = int((k + 1) / BUFFER_SIZE)
                    num_train += 1
                    if num_train % BUFFER_SIZE == 0:
                        tf.compat.v1.logging.info(
                            "Train samples are : {}".format(num_train))
                    if checker > shard:
                        print("TFRecord {} is completed.".format(prefix_train +
                                                                 str(shard)))
                        train_writers.close()

                visualization.percent(k, len(raw_data))
    else:
        for k, f in enumerate(raw_data):
            v = f + '.mp4'
            v_data, cnt = image_parser.get_raw_dataset(path=v)
            if len(v_data) > 0:
                w = f + '.txt'
                w = text_parser.one_file_encoder(w, 0)
                if len(v_data.shape) == 4:
                    v_data = tf.reshape(v_data, [-1])
                    if checker == shard:
                        pass
                    else:
                        shard = k // BUFFER_SIZE
                        train_writers = tf.io.TFRecordWriter(
                            prefix_train + str(index * 2000000 + shard),
                            options=options)
                    example = dict_to_example(v_data.numpy().tolist(), w, cnt)
                    train_writers.write(example.SerializeToString())
                    checker = int((k + 1) / BUFFER_SIZE)
                    num_train += 1
                    if num_train % BUFFER_SIZE == 0:
                        tf.compat.v1.logging.info(
                            "Train samples are : {}".format(num_train))
                    if checker > shard:
                        print("TFRecord {} is completed.".format(prefix_train +
                                                                 str(shard)))
                        train_writers.close()

                visualization.percent(k, len(raw_data))


ROOT_PATH = ''
TRAIN_PATH = ''
TFRecord_PATH = cwd + '/lip_reading_data'


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess TFRecord')
    parser.add_argument('-t',
                        '--text',
                        nargs='?')
    parser.add_argument('-v',
                        '--video',
                        nargs='?')
    args = parser.parse_args()
    text = args.text
    video = args.video
    global ROOT_PATH
    ROOT_PATH = video
    global TRAIN_PATH
    TRAIN_PATH = text
    files = get_file()
    tfrecord_generater(TFRecord_PATH, files, 1)


if __name__ == "__main__":
    main()
