import tensorflow as tf
import core_data_SRCandTGT
TRAIN_PATH = '/home/vivalavida/massive_data/lip_reading_data/sentence_level_lrs2/lr_train.txt'
ROOT_PATH = '/home/vivalavida/massive_data/lip_reading_data/sentence_level_lrs2/main'
# TRAIN_PATH = '/home/ubuntu/workspace/lr_train.txt'
# ROOT_PATH = '/home/ubuntu/workspace/main'
DATA_PATH = '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng'
sentenceHelper = core_data_SRCandTGT.DatasetManager(
    [DATA_PATH + "/europarl-v7.fr-en.en"],
    [DATA_PATH + "/europarl-v7.fr-en.en"],
    batch_size=16,
    shuffle=100)


def get_file():
    files = []
    with open(TRAIN_PATH, 'r') as f:
        files = f.readlines()
    files = [ROOT_PATH + '/' + f.rstrip() + '.txt' for f in files]
    print(files[:5])
    return files


files = get_file()
raw_data = []
b = 0
s = ''
for file in files:
    with tf.io.gfile.GFile(file, "r") as f:
        raw = f.readlines()
        a = len(sentenceHelper.encode(raw[0].lower()))
        b = max(a, b)
        if a == b:
            s = raw[0].lower()
        f.close()
print(b)
print(s)
with tf.io.gfile.GFile("lip_corpus_train.txt", "w") as f:
    for raw in raw_data:
        f.write(raw)
    f.close()
