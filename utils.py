
import collections
import tensorflow as tf
import numpy as np
import sys
import time

def load_data(file_dir):
  start_time = time.time()
  f = open(file_dir, 'r')
  sentences = []
  while True:
    sentence = f.readline()
    if not sentence:
      break

    sentence = sentence.strip()
    sentences.append(sentence)

  f.close()
  print_out("loaded %d sentences from files, time %.2fs" % (len(sentences), time.time() - start_time))
  return sentences

def build_vocab(sentences, max_words=None):
  word_count = collections.Counter()
  for sentence in sentences:
    for word in sentence.split():
      word_count[word] += 1

  print "the dataset has %d different words totally" % len(word_count)
  if not max_words:
    max_words = len(word_count)
  filter_out_words = len(word_count) - max_words
  word_dict = word_count.most_common(max_words)
  index2word = ["<s>"] + ["</s>"] + [word[0] for word in word_dict]
  word2index = dict([(word, idx) for idx, word in enumerate(index2word)])

  print "%d words filtered out of the vocabulary and %d words in the vocabulary" % (filter_out_words, len(index2word))
  return index2word, word2index

def save_vocab_list(file_dir, index2word):
  f = open(file_dir, "w")
  for word in index2word:
    f.write("".join(word) + "\n")
  f.close()

def build_vocab_from_file(vocab_file):
  f = open(vocab_file, 'r')
  index2word = f.readlines()
  index2word = map(lambda x: x.split('\t')[0], index2word)
  index2word = ["<unk>"] + ["<s>"] + ["</s>"] + index2word
  word2index = dict([(word, idx) for idx, word in enumerate(index2word)])
  f.close()
  print_out("%d words loadded from vocab file" % len(index2word))
  return index2word, word2index

def build_vocab_from_file_with_length(vocab_file, read_length):
  index2word, _ = build_vocab_from_file(vocab_file)
  index2word = index2word[: read_length]
  word2index = dict([(word, idx) for idx, word in enumerate(index2word)])
  return index2word, word2index

def load_vocab_from_file(vocab_file):
  f = open(vocab_file, 'r')
  index2word = f.readlines()
  index2word = map(lambda x: x.strip(), index2word)
  word2index = dict([(word, idx) for idx, word in enumerate(index2word)])
  f.close()
  print_out("%d words loadded from vocab file" % len(index2word))
  return index2word, word2index

def vectorize(data, word2index):
  vec_data = []
  for item in data:
    vec_review = [word2index[w] if w in word2index else 0 for w in item.review.split()]
    new_item = DataItem(user=int(item.user),
                        product=int(item.product),
                        rating=int(float(item.rating)),
                        review=vec_review)
    vec_data.append(new_item)
  return vec_data

def de_vectorize(sample_id, index2word):
  """ The reverse process of vectorization"""
  return " ".join([index2word[int(i)] for i in sample_id if i >= 0])

def padding_data(sentences):
  """
    in general,when padding data,first generate all-zero matrix,then for every
    sentence,0 to len(seq) assigned by seq,like pdata[idx, :lengths[idx]] = seq

      pdata: data after zero padding
      lengths: length of sentences
  """
  lengths = [len(s) for s in sentences]
  n_samples = len(sentences)
  max_len = np.max(lengths)
  pdata = np.zeros((n_samples, max_len)).astype('int32')
  for idx, seq in enumerate(sentences):
    pdata[idx, :lengths[idx]] = seq
  return pdata, lengths 

def get_batchidx(n_data, batch_size, shuffle=False):
  """
    batch all data index into a list
  """
  idx_list = np.arange(n_data)
  if shuffle:
    np.random.shuffle(idx_list)
  batch_index = []
  num_batches = int(np.ceil(float(n_data) / batch_size))
  for idx in range(num_batches):
    start_idx = idx * batch_size
    batch_index.append(idx_list[start_idx: min(start_idx + batch_size, n_data)])
  return batch_index

class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target_input",
                                           "target_output",
                                           "source_sequence_length",
                                           "target_sequence_length"))):
  pass

def get_iterator(dataset,
                 vocab_table,
                 batch_size,
                 random_seed,
                 shuffle=True,
                 source_reverse=False,
                 output_buffer_size=None):

  if not output_buffer_size:
    output_buffer_size = batch_size * 1000
  unk_id = tf.cast(vocab_table.lookup(tf.constant("<unk>")), tf.int32)
  sos_id = tf.cast(vocab_table.lookup(tf.constant("<s>")), tf.int32)
  eos_id = tf.cast(vocab_table.lookup(tf.constant("</s>")), tf.int32)

  if shuffle:
    dataset = dataset.shuffle(output_buffer_size, random_seed)

  src_tgt_dataset = dataset.map(
    lambda src: (tf.string_split([src]).values, tf.string_split([src]).values))

  if source_reverse:
    src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.reverse(src, axis=[0]), tgt))

  src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (tf.cast(vocab_table.lookup(src), tf.int32),
                      tf.cast(vocab_table.lookup(tgt), tf.int32)))

  src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (src,
                      tf.concat(([sos_id], tgt), 0),
                      tf.concat((tgt, [eos_id]), 0)))

  src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt_in, tgt_out: (src,
                                  tgt_in,
                                  tgt_out,
                                  tf.size(src),
                                  tf.size(tgt_in)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len

        padding_values=(
            unk_id,  # src
            unk_id,  # tgt_input
            unk_id,  # tgt_output
            0,  # src_len -- unused
            0))  # tgt_len -- unused

  batch_dataset = batching_func(src_tgt_dataset)
  batch_iterator = batch_dataset.make_initializable_iterator()
  (source, target_input, target_output, src_seq_len, tgt_seq_len) = (batch_iterator.get_next())
  return BatchedInput(
    initializer=batch_iterator.initializer,
    source=source,
    target_input=target_input,
    target_output=target_output,
    source_sequence_length=src_seq_len,
    target_sequence_length=tgt_seq_len)

def get_infer_iterator(dataset,
                       vocab_table,
                       batch_size,
                       source_reverse=False):
  unk_id = tf.cast(vocab_table.lookup(tf.constant("<unk>")), tf.int32)

  dataset = dataset.map(lambda src: tf.string_split([src]).values)
  dataset = dataset.map(lambda src: tf.cast(vocab_table.lookup(src), tf.int32))

  if source_reverse:
    dataset = dataset.map(lambda src: tf.reverse(src, axis=[0]))
  dataset = dataset.map(lambda src: (src, tf.size(src)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        padded_shapes=(
          tf.TensorShape([None]),
          tf.TensorShape([])),
        padding_values=(
          unk_id,
          0))

  batch_dataset = batching_func(dataset)
  batch_iterator = batch_dataset.make_initializable_iterator()
  (source, src_seq_len) = batch_iterator.get_next()
  return BatchedInput(
    initializer=batch_iterator.initializer,
    source=source,
    target_input=None,
    target_output=None,
    source_sequence_length=src_seq_len,
    target_sequence_length=None)

def sequence_accuracy(pred, truth):
  pred = pred[:-1]
  if len(pred) > len(truth):
    pred = pred[:len(truth)]
  elif len(pred) < len(truth):
    pred = pred + ["eos"] * (len(truth) - len(pred))

  true_words = 0.
  for idx, word in enumerate(pred):
    if word == truth[idx]:
      true_words += 1.
  return true_words / len(pred)

def print_time(s, start_time):
  """Take a start time, print elapsed duration, and return a new time."""
  print_out("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
  sys.stdout.flush()
  return time.time()

def print_out(s, f=None, new_line=True):
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  out_s = s.encode("utf-8")
  if not isinstance(out_s, str):
    out_s = out_s.decode("utf-8")
  print out_s,

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()

def format_text(words):
  """Convert a sequence words into sentence."""
  if (not hasattr(words, "__len__") and  # for numpy array
      not isinstance(words, collections.Iterable)):
    words = [words]
  return " ".join(words)
  # return b" ".join(words)

def get_tgt_sequence(outputs, sent_id, tgt_eos):
  if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
  # Select a sentence
  output = outputs[:, sent_id].tolist()

  # If there is an eos symbol in outputs, cut them at that point.
  if tgt_eos and tgt_eos in output:
    output = output[:output.index(tgt_eos)]
  translation = format_text(output)
  return translation

#tensorflow utils
def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto
