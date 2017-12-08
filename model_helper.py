
import tensorflow as tf
import numpy as np
import collections
import utils
import time

from tensorflow.python.ops import lookup_ops
from model import VRAE

class TrainModel(collections.namedtuple("TrainModel", ("graph", "model", "iterator"))):
  pass

def build_train_model(args, name="train_model", use_attention=False, scope=None):
  vocab_dir = args.vocab_dir
  data_dir = args.train_dir
  graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "train"):
    vocab_table = lookup_ops.index_table_from_file(vocab_dir, default_value=0)
    dataset = tf.data.TextLineDataset(data_dir)

    iterator = utils.get_iterator(
        dataset=dataset,
        vocab_table=vocab_table,
        batch_size=args.batch_size,
        random_seed=123,
        shuffle=True,
        source_reverse=False)

    if use_attention:
      pass
    else:
      model = VRAE(args, tf.contrib.learn.ModeKeys.TRAIN, iterator, vocab_table, name=name)
    return TrainModel(graph=graph, model=model, iterator=iterator)

class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model", "file_placeholder", "iterator"))):
  pass

def build_eval_model(args, name="eval_model", use_attention=False, scope=None):
  vocab_dir = args.vocab_dir
  graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "eval"):
    vocab_table = lookup_ops.index_table_from_file(vocab_dir, default_value=0)
    file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
    dataset = tf.data.TextLineDataset(file_placeholder)

    iterator = utils.get_iterator(
        dataset=dataset,
        vocab_table=vocab_table,
        batch_size=args.max_batch,
        random_seed=123,
        shuffle=True,
        source_reverse=False)

    if use_attention:
      pass
    else:
      model = VRAE(args, tf.contrib.learn.ModeKeys.EVAL, iterator, vocab_table, name=name)

  return EvalModel(graph=graph, 
                   model=model, 
                   file_placeholder=file_placeholder,
                   iterator=iterator)

class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model", "file_placeholder",
                            "batch_size_placeholder", "iterator"))):
  pass

def build_infer_model(args, name="infer_model", use_attention=False, scope=None):
  graph = tf.Graph()
  vocab_dir = args.vocab_dir

  with graph.as_default(), tf.container(scope or 'infer'):
    vocab_table = src_vocab_table = lookup_ops.index_table_from_file(
        vocab_dir, default_value=0)
    reverse_vocab_table = lookup_ops.index_to_string_table_from_file(
        vocab_dir, default_value="<unk>")

    if use_attention:
      pass
    else:
      model = VRAE(args, tf.contrib.learn.ModeKeys.INFER, 
          vocab_table, reverse_vocab_table, name=name)

    return InferModel(graph=graph, model=model)

def load_model(model, ckpt, session, name):
  start_time = time.time()
  model.saver.restore(session, ckpt)
  session.run(tf.tables_initializer())
  utils.print_out(
      "loaded %s model parameters from %s, time %.2fs" %
      (name, ckpt, time.time() - start_time))
  return model

def create_or_load_model(model, model_dir, session, name):
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt:
    model = load_model(model, latest_ckpt, session, name)
  else:
    start_time = time.time()
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    utils.print_out("created %s model with fresh parameters, time %.2fs" %
        (name, time.time() - start_time))

  global_step = model.global_step.eval(session=session)
  return model, global_step