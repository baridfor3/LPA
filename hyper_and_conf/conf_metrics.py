# encoder=utf8
import collections
import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from hyper_and_conf import hyper_util
import tensorflow as tf
import six
import unicodedata
import sys
import re
from jiwer import wer


def firstToken_loss(true, pred, mask_id=0, smoothing=0.1, vocab_size=1350):

    confidence = 1.0 - smoothing
    low_confidence = (1.0 - confidence) / tf.cast(vocab_size - 1, tf.float32)
    soft_targets = tf.one_hot(tf.cast(true[:, 0], tf.int32),
                              depth=vocab_size,
                              on_value=confidence,
                              off_value=low_confidence)
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred[:, 0],
                                                       labels=soft_targets)

    # Calculate the best (lowest) possible value of cross entropy, and
    # subtract from the cross entropy loss.
    normalizing_constant = -(confidence * tf.math.log(confidence) + tf.cast(
        vocab_size - 1, tf.float32) * low_confidence *
                             tf.math.log(low_confidence + 1e-20))
    xentropy -= normalizing_constant
    loss = tf.reduce_mean(input_tensor=xentropy)
    return loss


def onehot_loss_function(true, pred, mask_id=0, smoothing=0.1,
                         vocab_size=1350):
    """Short summary.
    Args:
        pred (type): Description of parameter `pred`.
        true (type): Description of parameter `true`.

    Returns:
        type: Description of returned object.

    """

    # mask = 1 - tf.cast(tf.equal(true, mask_id), tf.float32)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=pred, labels=true) * mask
    # return tf.reduce_mean(loss)
    logits, labels = hyper_util.pad_tensors_to_same_length(pred, true)
    # Calculate smoothing cross entropy
    with tf.name_scope("smoothing_cross_entropy"):
        confidence = 1.0 - smoothing
        low_confidence = (1.0 - confidence) / tf.cast(vocab_size - 1,
                                                      tf.float32)
        soft_targets = tf.one_hot(tf.cast(labels, tf.int32),
                                  depth=vocab_size,
                                  on_value=confidence,
                                  off_value=low_confidence)
        if len(logits.get_shape().as_list()) <= 2:
            logits = tf.one_hot(tf.cast(logits, tf.int32), depth=vocab_size)
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                           labels=soft_targets)

        # Calculate the best (lowest) possible value of cross entropy, and
        # subtract from the cross entropy loss.
        normalizing_constant = -(
            confidence * tf.math.log(confidence) +
            tf.cast(vocab_size - 1, tf.float32) * low_confidence *
            tf.math.log(low_confidence + 1e-20))
        xentropy -= normalizing_constant

    weights = tf.cast(tf.not_equal(labels, 0), tf.float32)

    weights = tf.cast(tf.not_equal(labels, mask_id), dtype=tf.float32)
    xentropy *= weights
    loss = tf.reduce_sum(input_tensor=xentropy) / tf.reduce_sum(
        input_tensor=weights)
    return loss


def token_trim(tokens, trim_id, remider=0):
    try:
        trim = tokens.index(trim_id) + int(remider)
        if trim == 0:
            tokens = tokens[:1]
        else:
            tokens = tokens[:trim]
    except Exception:
        tokens
    return tokens


def compute_wer(reference_corpus, translation_corpus, print_matrix=False):
    try:
        # eos_id = 1
        reference = reference_corpus.numpy().tolist()
        translation = translation_corpus.numpy().tolist()
    except Exception:
        # eos_id = 1
        reference = reference_corpus
        translation = translation_corpus
    # for (references, translations) in zip(reference_corpus,
    #                                       translation_corpus):
    score = 0
    num = 0
    for (ref, tra) in zip(reference, translation):
        hyp = token_trim(tra, 1, remider=1)
        # ref = token_trim(reference_corpus, 0)
        ref = token_trim(ref, 1, remider=1)
        # hyp = translation_corpus
        # ref = reference_corpus
        # N = len(hyp)
        # M = len(ref)
        # L = np.zeros((N, M))
        # for i in range(0, N):
        #     for j in range(0, M):
        #         if min(i, j) == 0:
        #             L[i, j] = max(i, j)
        #         else:
        #             deletion = L[i - 1, j] + 1
        #             insertion = L[i, j - 1] + 1
        #             sub = 1 if hyp[i] != ref[j] else 0
        #             substitution = L[i - 1, j - 1] + sub
        #             L[i, j] = min(deletion, min(insertion, substitution))
        s = wer(' '.join(str(r) for r in ref), ' '.join(str(h) for h in hyp))
        # print("{} - {}: del {} ins {} sub {} s {}".format(hyp[i], ref[j], deletion, insertion, substitution, sub))
        # if print_matrix:
        #     print("WER matrix ({}x{}): ".format(N, M))
        #     print(L)
        # score += float(int(L[N - 1, M - 1]) / M)
        score += s
        # print(float(int(L[N - 1, M - 1]) / M))
        num += 1
    # batch_score += score
    return score / num


def wer_score(labels, logits):
    if len(logits.get_shape().as_list()) > 2:
        logits = tf.argmax(logits, axis=-1)
    else:
        logits = tf.cast(logits, tf.int64)
    labels = tf.cast(labels, tf.int64)

    # data = tf.stack([logits, labels], axis=1)
    # data = tf.cast(data, dtype=tf.float32)
    #
    # def f(data):
    #     wer = tf.py_function(compute_wer, (data[0], data[1]), tf.float32)
    #     return wer
    #
    # wer = tf.map_fn(f, data)
    wer = tf.py_function(compute_wer, [labels, logits], tf.float32)
    return wer, tf.constant(1.)


def wer_fn(labels, logits):
    return wer_score(labels, logits)[0]


def _get_ngrams_with_counter(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(raw_reference_corpus,
                 raw_translation_corpus,
                 eos_id=1,
                 max_order=4,
                 use_bp=True):
    """Computes BLEU score of translated segments against one or more references.
  Args:
    reference_corpus: list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.
  Returns:
    BLEU score.
  """
    try:
        # eos_id = eos_id.numpy()
        # eos_id = 1
        reference_corpus = raw_reference_corpus.numpy().tolist()
        translation_corpus = raw_translation_corpus.numpy().tolist()
    except Exception:
        eos_id = eos_id
        # eos_id = 1
        reference_corpus = raw_reference_corpus
        translation_corpus = raw_translation_corpus
    num = 0
    bleu = 0
    for (references, translations) in zip(reference_corpus,
                                          translation_corpus):
        reference_length = 0
        translation_length = 0
        bp = 1.0
        geo_mean = 0

        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        precisions = []
        references = token_trim(references, 1, remider=1)
        translations = token_trim(translations, 1, remider=1)
        # references = data_manager.decode(references).split(' ')
        # translations = data_manager.decode(translations).split(' ')
        reference_length += len(references)
        translation_length += len(translations)
        ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
        translation_ngram_counts = _get_ngrams_with_counter(
            translations, max_order)

        overlap = dict((ngram, min(count, translation_ngram_counts[ngram]))
                       for ngram, count in ref_ngram_counts.items())

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram) -
                                      1] += translation_ngram_counts[ngram]

        precisions = [0] * max_order
        smooth = 1.0

        for i in xrange(0, max_order):
            if possible_matches_by_order[i] > 0:
                precisions[i] = float(
                    matches_by_order[i]) / possible_matches_by_order[i]
                if matches_by_order[i] > 0:
                    precisions[i] = float(
                        matches_by_order[i]) / possible_matches_by_order[i]
                else:
                    smooth *= 2
                    precisions[i] = 1.0 / (smooth *
                                           possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

        if max(precisions) > 0:
            p_log_sum = sum(math.log(p) for p in precisions if p)
            geo_mean = math.exp(p_log_sum / max_order)

        if use_bp:
            ratio = translation_length / reference_length
            bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
        bleu += geo_mean * bp
        num += 1
    return np.float32(bleu / num), tf.constant(1.)


def approx_bleu(labels, logits):
    if len(logits.get_shape().as_list()) > 2:
        logits = tf.argmax(logits, axis=-1)
    else:
        logits = tf.cast(logits, tf.int64)
    labels = tf.cast(labels, tf.int64)
    # num = tf.shape(logits)[0]
    # data = tf.stack([labels, logits], axis=1)
    # data = tf.cast(data, dtype=tf.float32)
    #
    # def f(data):
    #     bleu = tf.py_function(compute_bleu, (data[0], data[1]), tf.float32)
    #     return bleu * 100
    #
    # score = tf.map_fn(f, data)
    score = tf.py_function(compute_bleu, [
        labels,
        logits,
    ], tf.float32)
    return score * 100, tf.constant(1.)


def approx_unigram_bleu(labels, logits):
    if len(logits.get_shape().as_list()) > 2:
        logits = tf.argmax(logits, axis=-1)
    else:
        logits = tf.cast(logits, tf.int64)
    labels = tf.cast(labels, tf.int64)
    # num = tf.shape(logits)[0]
    # data = tf.stack([labels, logits], axis=1)
    # data = tf.cast(data, dtype=tf.float32)
    #
    # def f(data):
    #     bleu = tf.py_function(compute_bleu, (data[0], data[1]), tf.float32)
    #     return bleu * 100
    #
    # score = tf.map_fn(f, data)
    score = tf.py_function(compute_unigram_bleu, [labels, logits], tf.float32)
    return score * 100, tf.constant(1.)


def compute_unigram_bleu(labels, logits):
    return compute_bleu(labels, logits, max_order=1)


def unigram_bleu_fn(labels, logits):
    return approx_unigram_bleu(labels, logits)[0]


def bleu_fn(labels, logits):
    return compute_bleu(labels, logits)[0]


class UnicodeRegex(object):
    """Ad-hoc hack to recognize all punctuation and symbols."""
    def __init__(self):
        punctuation = self.property_chars("P")
        self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
        self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
        self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

    def property_chars(self, prefix):
        return "".join(
            six.unichr(x) for x in range(sys.maxunicode)
            if unicodedata.category(six.unichr(x)).startswith(prefix))


uregex = UnicodeRegex()


def bleu_tokenize(string):
    r"""Tokenize a string following the official BLEU implementation.
  See https://github.com/moses-smt/mosesdecoder/'
           'blob/master/scripts/generic/mteval-v14.pl#L954-L983
  In our case, the input string is expected to be just one line
  and no HTML entities de-escaping is needed.
  So we just tokenize on punctuation and symbols,
  except when a punctuation is preceded and followed by a digit
  (e.g. a comma/dot as a thousand/decimal separator).
  Note that a numer (e.g. a year) followed by a dot at the end of sentence
  is NOT tokenized,
  i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
  does not match this case (unless we add a space after each sentence).
  However, this error is already in the original mteval-v14.pl
  and we want to be consistent with it.
  Args:
    string: the input string
  Returns:
    a list of tokens
  """
    string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
    string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
    string = uregex.symbol_re.sub(r" \1 ", string)
    return string.split()


def bleu_wrapper(ref_filename, hyp_filename, case_sensitive=False):
    """Compute BLEU for two files (reference and hypothesis translation)."""
    ref_lines = tf.io.gfile.GFile(ref_filename).read().strip().splitlines()
    hyp_lines = tf.io.gfile.GFile(hyp_filename).read().strip().splitlines()
    # ref_lines = ['I like cat']
    # hyp_lines = ['I like dog']

    if len(ref_lines) != len(hyp_lines):
        raise ValueError(
            "Reference and translation files have different number of "
            "lines.")
    if not case_sensitive:
        ref_lines = [x.lower() for x in ref_lines]
        hyp_lines = [x.lower() for x in hyp_lines]
    ref_tokens = [bleu_tokenize(x) for x in ref_lines]
    hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]
    return compute_bleu(ref_tokens, hyp_tokens) * 100


# sess = tf.Session()
# re = tf.convert_to_tensor([[2, 2, 3, 1, 0, 0], [3, 4, 6, 1, 0,
#                                                 0]]).eval(session=sess)
# tr = [[1, 2, 2, 134, 234, 123, 3, 2], [3, 6, 61, 23, 5, 6, 7, 2]]
# s, c = bleu_score(tr, re, 1)
# c.eval(session=sess)
# s.eval(session=sess)
# ref = '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng/europarl-v7.fr-en.en_test'
# tra = '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng/europarl-v7.fr-en.fr_test'
# bleu_wrapper(ref, tra)


def padded_accuracy(labels, logits):
    """Percentage of times that predictions matches labels on non-0s."""
    with tf.name_scope("padded_accuracy"):
        logits, labels = hyper_util.pad_tensors_to_same_length(logits, labels)
        weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
        outputs = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        padded_labels = tf.cast(labels, tf.int32)
        return tf.cast(tf.equal(outputs, padded_labels), tf.float32), weights


def padded_accuracy_topk(labels, logits, k):
    """Percentage of times that top-k predictions matches labels on non-0s."""
    with tf.name_scope("padded_accuracy_topk"):
        logits, labels = hyper_util.pad_tensors_to_same_length(logits, labels)
        weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
        effective_k = tf.minimum(k, tf.shape(logits)[-1])
        _, outputs = tf.nn.top_k(logits, k=effective_k)
        outputs = tf.cast(outputs, tf.int32)
        padded_labels = tf.cast(labels, tf.int32)
        padded_labels = tf.expand_dims(padded_labels, axis=-1)
        padded_labels += tf.zeros_like(outputs)  # Pad to same shape.
        same = tf.cast(tf.equal(outputs, padded_labels), tf.float32)
        same_topk = tf.reduce_sum(same, axis=-1)
        return same_topk, weights


def padded_accuracy_top5(labels, logits):
    return padded_accuracy_topk(labels, logits, 5)


def padded_accuracy_top1(labels, logits):
    return padded_accuracy_topk(labels, logits, 1)


def padded_sequence_accuracy(labels, logits):
    """Percentage of times that predictions matches labels everywhere (non-0)."""
    with tf.name_scope("padded_sequence_accuracy"):
        logits, labels = hyper_util.pad_tensors_to_same_length(logits, labels)
        weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
        outputs = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        padded_labels = tf.cast(labels, tf.int32)
        not_correct = tf.cast(tf.not_equal(outputs, padded_labels),
                              tf.float32) * weights
        axis = list(range(1, len(outputs.get_shape())))
        correct_seq = 1.0 - tf.minimum(1.0,
                                       tf.reduce_sum(not_correct, axis=axis))
        return correct_seq, tf.constant(1.0)


# a = tf.convert_to_tensor([[2, 2, 3, 2,2,1, 0, 0], [2, 2, 3, 2,2,1, 0, 0]])
# b = tf.convert_to_tensor([[2, 3, 3, 1, 0, 0],[2, 2, 3, 1, 0, 0]])
# # a = tf.convert_to_tensor([[2], [2]])
# #
# # b = tf.convert_to_tensor([[3], [2]])
# c = tf.convert_to_tensor([
#     [2, 2, 2, 134, 234, 123, 3, 2],
# ])
# d = tf.convert_to_tensor([[3, 6, 61, 23, 5, 6, 7, 2]])
# re = tf.convert_to_tensor([[2, 2, 3, 1, 0, 0], [3, 4, 6, 1, 0, 0]])
# tr = tf.convert_to_tensor([[2, 2, 2, 134, 234, 123, 3, 2],
#                            [3, 6, 61, 23, 5, 6, 7, 2]])
#
# bleu, c = approx_bleu(a, b)
# bleu
# s, c = wer_score(a, b)
# print(s)
# s
# p,_ = padded_accuracy(a,b)
