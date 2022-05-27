# encoding=utf8
import tensorflow as tf

import collections
import re
import sys
import unicodedata
import numpy as np
import six
# import tensorflow as tf

# Set of characters that will be used in the function _escape_token() (see func
# docstring for more details).
# This set is added to the alphabet list to ensure that all escaped tokens can
# be encoded.
_ESCAPE_CHARS = set(u"\\_u;0123456789")
# Regex for the function _unescape_token(), the inverse of _escape_token().
# This is used to find "\u", "\\", and "\###;" substrings in the token.
_UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")

_UNDEFINED_UNICODE = u"\u3013"

# Set contains all letter and number characters.
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L")
        or unicodedata.category(six.unichr(i)).startswith("N")))
# min_count is the minimum number of times a subtoken must appear in the data
# before before it is added to the vocabulary. The value is found using binary
# search to obtain the target vocabulary size.
_MIN_MIN_COUNT = 1  # min value to use when binary searching for min_count
_MAX_MIN_COUNT = 200000  # max value to use when binary searching for min_count


class Subtokenizer(object):
    """Encodes and decodes strings to/from integer IDs."""

    def __init__(self,
                 vocab_file,
                 files=None,
                 vocabulary_size=42,
                 PAD_ID=0,
                 EOS_ID=1,
                 PAD='<PADDDD>',
                 EOS='<EOSSSS>',
                 MASK='<MASK>',
                 reserved_tokens=None):
        """Initializes class, creating a vocab file if data_files is provided."""
        print("Initializing Subtokenizer from file %s." % files)
        self.PAD_ID = PAD_ID
        self.EOS_ID = EOS_ID
        self.MASK = MASK
        self.RESERVED_TOKENS = [PAD, EOS, MASK]
        if reserved_tokens is None:
            reserved_tokens = self.RESERVED_TOKENS
        vocab_file = vocab_file
        if tf.io.gfile.exists(vocab_file):
            print("Vocab file already exists (%s)" % vocab_file)
            self.subtoken_list = _load_vocab_file(
                vocab_file, reserved_tokens=self.RESERVED_TOKENS)
            self.alphabet = _generate_alphabet_dict(
                self.subtoken_list, reserved_tokens=self.RESERVED_TOKENS)
            self.subtoken_to_id_dict = _list_to_index_dict(self.subtoken_list)
        else:
            # vocabulary = _generate_WORD_LEVEL_vocabulary(file)
            # vocabulary_size = len(vocabulary)
            threshold = int(vocabulary_size * 0.01)
            print("Building vocab file from (%s)" % vocab_file)
            self.init_from_files(vocab_file, files, vocabulary_size, threshold)
        self.max_subtoken_length = 0
        for subtoken in self.subtoken_list:
            self.max_subtoken_length = max(self.max_subtoken_length,
                                           len(subtoken))

        # Create cache to speed up subtokenization

        self._cache_size = 2**20
        self._cache = [(None, None)] * self._cache_size

    def init_from_files(self,
                        vocab_file,
                        files,
                        target_vocab_size,
                        threshold,
                        min_count=None,
                        file_byte_limit=1e6,
                        reserved_tokens=None):
        """Create subtoken vocabulary based on files, and save vocab to file.
    Args:
      vocab_file: String name of vocab file to store subtoken vocabulary.
      files: List of file paths that will be used to generate vocabulary.
      target_vocab_size: target vocabulary size to generate.
      threshold: int threshold of vocabulary size to accept.
      min_count: int minimum count to use for generating the vocabulary. The min
        count is the minimum number of times a subtoken should appear in the
        files before it is added to the vocabulary. If set to none, this value
        is found using binary search.
      file_byte_limit: (Default 1e6) Maximum number of bytes of sample text that
        will be drawn from the files.
      reserved_tokens: List of string tokens that are guaranteed to be at the
        beginning of the subtoken vocabulary list.
    Returns:
      Subtokenizer object
    """
        if reserved_tokens is None:
            reserved_tokens = reserved_tokens

        print("Begin steps to create subtoken vocabulary...")
        token_counts = _count_tokens(files, file_byte_limit)
        self.vocab_size = token_counts
        alphabet = _generate_alphabet_dict(
            token_counts, reserved_tokens=self.RESERVED_TOKENS)
        subtoken_list = _generate_subtokens_with_target_vocab_size(
            token_counts,
            alphabet,
            target_vocab_size,
            threshold,
            min_count,
            reserved_tokens=self.RESERVED_TOKENS)
        print("Generated vocabulary with %d subtokens." % len(subtoken_list))
        _save_vocab_file(vocab_file, subtoken_list)
        self.subtoken_list = _load_vocab_file(
            vocab_file, reserved_tokens=self.RESERVED_TOKENS)
        self.alphabet = _generate_alphabet_dict(self.subtoken_list,
                                                self.RESERVED_TOKENS)
        self.subtoken_to_id_dict = _list_to_index_dict(self.subtoken_list)
    def truncated_encode(self, raw_string, add_eos=True):
        """Encodes a string into a list of int subtoken ids."""
        ret = []
        if isinstance(raw_string, np.ndarray):
            # Note that list(subtokens) converts subtokens to a python list, but the
            # items remain as np.int32. This converts both the array and its items.
            pass
        else:
            raw_string = raw_string.numpy()
        raw_string = tf.compat.as_str_any(raw_string)
        # raw_string = raw_string[:-
        tokens = _split_string_to_tokens(raw_string)
        for token in tokens:
            ret.extend(self._token_to_subtoken_ids(token))
        if add_eos:
            ret.append(self.EOS_ID)
        return ret

    def encode(self, raw_string, add_eos=True):
        """Encodes a string into a list of int subtoken ids."""
        ret = []
        # if isinstance(raw_string, np.ndarray):
        #     # Note that list(subtokens) converts subtokens to a python list, but the
        #     # items remain as np.int32. This converts both the array and its items.
        #     pass
        # else:
        #     raw_string = raw_string.numpy()
        # raw_string = raw_string.decode('utf-8')
        tokens = _split_string_to_tokens(raw_string)
        for token in tokens:
            ret.extend(self._token_to_subtoken_ids(token))
        if add_eos:
            ret.append(self.EOS_ID)
        return ret

    def _token_to_subtoken_ids(self, token):
        """Encode a single token into a list of subtoken ids."""
        cache_location = hash(token) % self._cache_size
        cache_key, cache_value = self._cache[cache_location]
        if cache_key == token:
            return cache_value

        ret = _split_token_to_subtokens(
            _escape_token(token, self.alphabet), self.subtoken_to_id_dict,
            self.max_subtoken_length)
        ret = [self.subtoken_to_id_dict[subtoken_id] for subtoken_id in ret]

        self._cache[cache_location] = (token, ret)
        return ret

    def decode(self, subtokens):
        """Converts list of int subtokens ids into a string."""
        if isinstance(subtokens, np.ndarray):
            # Note that list(subtokens) converts subtokens to a python list, but the
            # items remain as np.int32. This converts both the array and its items.
            subtokens = subtokens.tolist()

        if not np.all(subtokens):
            return ""
        # assert not (
        #     isinstance(subtokens, list) and isinstance(subtokens[0], int)
        # ), ("Subtokens argument passed into decode() must be a list of integers."
        #     )

        return _join_tokens_to_string(self._subtoken_ids_to_tokens(subtokens))

    def _subtoken_ids_to_tokens(self, subtokens):
        """Convert list of int subtoken ids to a list of string tokens."""
        escaped_tokens = "".join([
            self.subtoken_list[s] for s in subtokens
            if s < len(self.subtoken_list)
        ])
        escaped_tokens = escaped_tokens.split("_")

        # All tokens in the vocabulary list have been escaped (see _escape_token())
        # so each token must be unescaped when decoding.
        ret = []
        for token in escaped_tokens:
            if token:
                ret.append(_unescape_token(token))
        return ret

    def vocabulary_size(self):
        return self.vocab_size


def _save_vocab_file(vocab_file, subtoken_list):
    """Save subtokens to file."""
    with tf.io.gfile.GFile(vocab_file, mode="w") as f:
        for subtoken in subtoken_list:
            f.write("'%s'\n" % subtoken)


def _load_vocab_file(vocab_file, reserved_tokens=None):
    """Load vocabulary while ensuring reserved tokens are at the top."""
    if reserved_tokens is None:
        reserved_tokens = reserved_tokens

    subtoken_list = []
    with tf.io.gfile.GFile(vocab_file, mode="r") as f:
        for line in f:
            subtoken = line.strip()
            subtoken = subtoken[1:-1]  # Remove surrounding single-quotes
            if subtoken in reserved_tokens:
                continue
            subtoken_list.append(subtoken)
    return reserved_tokens + subtoken_list


def _split_string_to_tokens(text):
    """Splits text to a list of string tokens."""
    if not text:
        return []
    ret = []
    token_start = 0
    # Classify each character in the input string
    is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
    for pos in range(1, len(text)):
        if is_alnum[pos] != is_alnum[pos - 1]:
            token = text[token_start:pos]
            if token != u" " or token_start == 0:
                ret.append(token)
            token_start = pos
    final_token = text[token_start:]
    ret.append(final_token)
    return ret


def _join_tokens_to_string(tokens):
    """Join a list of string tokens into a single string."""
    token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in tokens]
    ret = []
    for i, token in enumerate(tokens):
        if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
            ret.append(u" ")
        ret.append(token)
    return "".join(ret)


def _escape_token(token, alphabet):
    r"""Replace characters that aren't in the alphabet and append "_" to token.
  Apply three transformations to the token:
    1. Replace underline character "_" with "\u", and backslash "\" with "\\".
    2. Replace characters outside of the alphabet with "\###;", where ### is the
       character's Unicode code point.
    3. Appends "_" to mark the end of a token.
  Args:
    token: unicode string to be escaped
    alphabet: list of all known characters
  Returns:
    escaped string
  """
    token = token.replace(u"\\", u"\\\\").replace(u"_", u"\\u")
    ret = [
        c if c in alphabet and c != u"\n" else r"\%d;" % ord(c) for c in token
    ]
    return u"".join(ret) + "_"


def _unescape_token(token):
    r"""Replaces escaped characters in the token with their unescaped versions.
  Applies inverse transformations as _escape_token():
    1. Replace "\u" with "_", and "\\" with "\".
    2. Replace "\###;" with the unicode character the ### refers to.
  Args:
    token: escaped string
  Returns:
    unescaped string
  """

    def match(m):
        r"""Returns replacement string for matched object.
    Matched objects contain one of the strings that matches the regex pattern:
      r"\\u|\\\\|\\([0-9]+);"
    The strings can be '\u', '\\', or '\###;' (### is any digit number).
    m.group(0) refers to the entire matched string ('\u', '\\', or '\###;').
    m.group(1) refers to the first parenthesized subgroup ('###').
    m.group(0) exists for all match objects, while m.group(1) exists only for
    the string '\###;'.
    This function looks to see if m.group(1) exists. If it doesn't, then the
    matched string must be '\u' or '\\' . In this case, the corresponding
    replacement ('_' and '\') are returned. Note that in python, a single
    backslash is written as '\\', and double backslash as '\\\\'.
    If m.goup(1) exists, then use the integer in m.group(1) to return a
    unicode character.
    Args:
      m: match object
    Returns:
      String to replace matched object with.
    """
        # Check if the matched strings are '\u' or '\\'.
        if m.group(1) is None:
            return u"_" if m.group(0) == u"\\u" else u"\\"

        # If m.group(1) exists, try and return unicode character.
        try:
            return six.unichr(int(m.group(1)))
        except (ValueError, OverflowError) as _:
            return _UNDEFINED_UNICODE

    # Use match function to replace escaped substrings in the token.
    return _UNESCAPE_REGEX.sub(match, token)


def _count_tokens(files, file_byte_limit=1e6):
    """Return token counts of words in the files.
  Samples file_byte_limit bytes from each file, and counts the words that appear
  in the samples. The samples are semi-evenly distributed across the file.
  Args:
    files: List of filepaths
    file_byte_limit: Max number of bytes that will be read from each file.
  Returns:
    Dictionary mapping tokens to the number of times they appear in the sampled
    lines from the files.
  """
    token_counts = collections.defaultdict(int)

    for file in files:
        with tf.io.gfile.GFile(file, mode="r") as reader:
            file_byte_budget = file_byte_limit
            counter = 0
            lines_to_skip = int(reader.size() / (file_byte_budget * 2))
            for line in reader:
                if counter < lines_to_skip:
                    counter += 1
                else:
                    if file_byte_budget < 0:
                        break
                    line = line.strip()
                    file_byte_budget -= len(line)
                    counter = 0

                    # Add words to token counts
                    for token in _split_string_to_tokens(line):
                        token_counts[token] += 1
    return token_counts


def _list_to_index_dict(lst):
    """Create dictionary mapping list items to their indices in the list."""
    return {item: n for n, item in enumerate(lst)}


def _split_token_to_subtokens(token, subtoken_dict, max_subtoken_length):
    """Splits a token into subtokens defined in the subtoken dict."""
    ret = []
    start = 0
    token_len = len(token)
    while start < token_len:
        # Find the longest subtoken, so iterate backwards.
        for end in range(
                min(token_len, start + max_subtoken_length), start, -1):
            subtoken = token[start:end]
            if subtoken in subtoken_dict:
                ret.append(subtoken)
                start = end
                break
        else:  # Did not break
            # If there is no possible encoding of the escaped token then one of the
            # characters in the token is not in the alphabet. This should be
            # impossible and would be indicative of a bug.
            raise ValueError(
                "Was unable to split token \"%s\" into subtokens." % token)
    return ret


def _generate_subtokens_with_target_vocab_size(token_counts,
                                               alphabet,
                                               target_size,
                                               threshold,
                                               min_count=None,
                                               reserved_tokens=None):
    """Generate subtoken vocabulary close to the target size."""
    if reserved_tokens is None:
        reserved_tokens = reserved_tokens

    if min_count is not None:
        print("Using min_count=%d to generate vocab with target size %d" %
              (min_count, target_size))
        return _generate_subtokens(
            token_counts, alphabet, min_count, reserved_tokens=reserved_tokens)

    def bisect(min_val, max_val):
        """Recursive function to binary search for subtoken vocabulary."""
        cur_count = (min_val + max_val) // 2
        print("Binary search: trying min_count=%d (%d %d)" %
              (cur_count, min_val, max_val))
        subtoken_list = _generate_subtokens(
            token_counts, alphabet, cur_count, reserved_tokens=reserved_tokens)

        val = len(subtoken_list)
        print("Binary search: min_count=%d resulted in %d tokens" % (cur_count,
                                                                     val))

        within_threshold = abs(val - target_size) < threshold
        if within_threshold or min_val >= max_val or cur_count < 2:
            return subtoken_list
        if val > target_size:
            other_subtoken_list = bisect(cur_count + 1, max_val)
        else:
            other_subtoken_list = bisect(min_val, cur_count - 1)

        # Return vocabulary dictionary with the closest number of tokens.
        other_val = len(other_subtoken_list)
        if abs(other_val - target_size) < abs(val - target_size):
            return other_subtoken_list
        return subtoken_list

    print("Finding best min_count to get target size of %d" % target_size)
    return bisect(_MIN_MIN_COUNT, _MAX_MIN_COUNT)


def _generate_alphabet_dict(iterable, reserved_tokens=None):
    """Create set of characters that appear in any element in the iterable."""
    if reserved_tokens is None:
        reserved_tokens = reserved_tokens
    alphabet = {c for token in iterable for c in token}
    alphabet |= {c for token in reserved_tokens for c in token}
    alphabet |= _ESCAPE_CHARS  # Add escape characters to alphabet set.
    return alphabet


def _count_and_gen_subtokens(token_counts, alphabet, subtoken_dict,
                             max_subtoken_length):
    """Count number of times subtokens appear, and generate new subtokens.
  Args:
    token_counts: dict mapping tokens to the number of times they appear in the
      original files.
    alphabet: list of allowed characters. Used to escape the tokens, which
      guarantees that all tokens can be split into subtokens.
    subtoken_dict: dict mapping subtokens to ids.
    max_subtoken_length: maximum length of subtoken in subtoken_dict.
  Returns:
    A defaultdict mapping subtokens to the number of times they appear in the
    tokens. The dict may contain new subtokens.
  """
    subtoken_counts = collections.defaultdict(int)
    for token, count in six.iteritems(token_counts):
        token = _escape_token(token, alphabet)
        subtokens = _split_token_to_subtokens(token, subtoken_dict,
                                              max_subtoken_length)

        # Generate new subtokens by taking substrings from token.
        start = 0
        for subtoken in subtokens:
            for end in range(start + 1, len(token) + 1):
                new_subtoken = token[start:end]
                subtoken_counts[new_subtoken] += count
            start += len(subtoken)

    return subtoken_counts


def _filter_and_bucket_subtokens(subtoken_counts, min_count):
    """Return a bucketed list of subtokens that are filtered by count.
  Args:
    subtoken_counts: defaultdict mapping subtokens to their counts
    min_count: int count used to filter subtokens
  Returns:
    List of subtoken sets, where subtokens in set i have the same length=i.
  """
    # Create list of buckets, where subtokens in bucket i have length i.
    subtoken_buckets = []
    for subtoken, count in six.iteritems(subtoken_counts):
        if count < min_count:  # Filter out subtokens that don't appear enough
            continue
        while len(subtoken_buckets) <= len(subtoken):
            subtoken_buckets.append(set())
        subtoken_buckets[len(subtoken)].add(subtoken)
    return subtoken_buckets


def _gen_new_subtoken_list(subtoken_counts,
                           min_count,
                           alphabet,
                           reserved_tokens=None):
    """Generate candidate subtokens ordered by count, and new max subtoken length.
  Add subtokens to the candiate list in order of length (longest subtokens
  first). When a subtoken is added, the counts of each of its prefixes are
  decreased. Prefixes that don't appear much outside the subtoken are not added
  to the candidate list.
  For example:
    subtoken being added to candidate list: 'translate'
    subtoken_counts: {'translate':10, 't':40, 'tr':16, 'tra':12, ...}
    min_count: 5
  When 'translate' is added, subtoken_counts is updated to:
    {'translate':0, 't':30, 'tr':6, 'tra': 2, ...}
  The subtoken 'tra' will not be added to the candidate list, because it appears
  twice (less than min_count) outside of 'translate'.
  Args:
    subtoken_counts: defaultdict mapping str subtokens to int counts
    min_count: int minumum count requirement for subtokens
    alphabet: set of characters. Each character is added to the subtoken list to
      guarantee that all tokens can be encoded.
    reserved_tokens: list of tokens that will be added to the beginning of the
      returned subtoken list.
  Returns:
    List of candidate subtokens in decreasing count order, and maximum subtoken
    length
  """
    if reserved_tokens is None:
        reserved_tokens = reserved_tokens

    # Create a list of (count, subtoken) for each candidate subtoken.
    subtoken_candidates = []

    # Use bucketted list to iterate through subtokens in order of length.
    # subtoken_buckets[i] = set(subtokens), where each subtoken has length i.
    subtoken_buckets = _filter_and_bucket_subtokens(subtoken_counts, min_count)
    max_subtoken_length = len(subtoken_buckets) - 1

    # Go through the list in reverse order to consider longer subtokens first.
    for subtoken_len in range(max_subtoken_length, 0, -1):
        for subtoken in subtoken_buckets[subtoken_len]:
            count = subtoken_counts[subtoken]

            # Possible if this subtoken is a prefix of another token.
            if count < min_count:
                continue

            # Ignore alphabet/reserved tokens, which will be added manually later.
            if subtoken not in alphabet and subtoken not in reserved_tokens:
                subtoken_candidates.append((count, subtoken))

            # Decrement count of the subtoken's prefixes (if a longer subtoken is
            # added, its prefixes lose priority to be added).
            for end in range(1, subtoken_len):
                subtoken_counts[subtoken[:end]] -= count

    # Add alphabet subtokens (guarantees that all strings are encodable).
    subtoken_candidates.extend(
        (subtoken_counts.get(a, 0), a) for a in alphabet)

    # Order subtoken candidates by decreasing count.
    subtoken_list = [t for _, t in sorted(subtoken_candidates, reverse=True)]

    # Add reserved tokens to beginning of the list.
    subtoken_list = reserved_tokens + subtoken_list
    return subtoken_list, max_subtoken_length


def _generate_WORD_LEVEL_vocabulary(file, line_token='\n', word_token=' '):
    def unicode_to_ascii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(w):
        w = unicode_to_ascii(w.lower().strip())

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

        w = w.rstrip().strip()

        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        return w

    print("Generating WORD_LEVEL vocabulary from %s" % file)
    lines = open(file, encoding='UTF-8').read().strip().split(line_token)

    sentences = [preprocess_sentence(l) for l in lines]
    vocab = set()
    for phrase in sentences:
        vocab.update(phrase.split(' '))
    with tf.io.gfile.GFile(file + '_WORD_LEVEL_vocabulary', mode="w") as f:
        for subtoken in vocab:
            f.write("'%s'\n" % subtoken)

    print("Generating WORD_LEVEL vocabulary finished, found %d words" %
          len(vocab))
    return vocab


def _generate_subtokens(token_counts,
                        alphabet,
                        min_count,
                        num_iterations=4,
                        reserved_tokens=None):
    """Create a list of subtokens in decreasing order of frequency.
  Args:
    token_counts: dict mapping str tokens -> int count
    alphabet: set of characters
    min_count: int minimum number of times a subtoken must appear before it is
      added to the vocabulary.
    num_iterations: int number of iterations to generate new tokens.
    reserved_tokens: list of tokens that will be added to the beginning to the
      returned subtoken list.
  Returns:
    Sorted list of subtokens (most frequent first)
  """
    if reserved_tokens is None:
        reserved_tokens = reserved_tokens

    # Use alphabet set to create initial list of subtokens
    subtoken_list = reserved_tokens + list(alphabet)
    max_subtoken_length = 1

    # On each iteration, segment all words using the subtokens defined in
    # subtoken_dict, count how often the resulting subtokens appear, and update
    # the dictionary with subtokens w/ high enough counts.
    for i in range(num_iterations):
        print("\tGenerating subtokens: iteration %d" % i)
        # Generate new subtoken->id dictionary using the new subtoken list.
        subtoken_dict = _list_to_index_dict(subtoken_list)

        # Create dict mapping subtoken->count, with additional subtokens created
        # from substrings taken from the tokens.
        subtoken_counts = _count_and_gen_subtokens(
            token_counts, alphabet, subtoken_dict, max_subtoken_length)

        # Generate new list of subtokens sorted by subtoken count.
        subtoken_list, max_subtoken_length = _gen_new_subtoken_list(
            subtoken_counts, min_count, alphabet, reserved_tokens)

        print("\tVocab size: %d" % len(subtoken_list))
    return subtoken_list


class Subtokenizer_convert(object):
    """Encodes and decodes strings to/from integer IDs."""

    def __init__(self, file_src, file_tgt):
        self.src = Subtokenizer(file_src)
        self.tgt = Subtokenizer(file_tgt)

    def covert(self, input):
        # input = input.numpy()
        # import pdb; pdb.set_trace()
        src = self.src.decode(input)[:-8]
        src = self.tgt.encode(src)
        return np.array(src, np.int64)


# import os
# cwd = os.getcwd()
#
# DIC_SRC = cwd + '/data/_BYTE_LEVEL_vocabulary_13k'
# DIC_TGT = cwd + '/data/_BYTE_LEVEL_vocabulary_1k3'
#
# Char = cwd + '/_BYTE_LEVEL_vocabulary_char'
# byter = Subtokenizer(Char, ['../corpus/lip_corpus.txt'])
# b = "faked"
# c = byter.encode(b)
# c
# dict_conver = Subtokenizer_convert(DIC_SRC, DIC_TGT)
# a = [123]
# dict_conver.covert(c)
