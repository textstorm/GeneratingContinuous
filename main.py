
import tensorflow as tf
import numpy as np
import utils
import config
import model_helper

import time
import random
import os
import nltk

def run_sample_decode(args, infer_model, infer_sess, model_dir, infer_data):
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer")

  _sample_decode(args, loaded_infer_model, global_step, infer_sess, infer_model.iterator, 
      infer_data, infer_model.data_placeholder, infer_model.batch_size_placeholder)

def _sample_decode(args, model, global_step, sess, iterator, infer_data, 
                   iterator_data_placeholder, iterator_batch_size_placeholder):
  decode_id = random.randint(0, len(infer_data) - 1)
  iterator_feed_dict = {
      iterator_data_placeholder: [infer_data[decode_id]],
      iterator_batch_size_placeholder: 1,
  }
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
  sample_id, sample_words = model.infer(sess)

  if args.beam_width > 0:
    sample_words = sample_words[0]

  utils.print_out("src sentence: %s" % infer_data[decode_id])
  utils.print_out("gen sentence: %s" % (" ".join(sample_words)))

def run_internal_eval(args, eval_model, eval_sess, model_dir, eval_data):
  with eval_model.graph.as_default():
    loaded_eval_model, global_step = model_helper.create_or_load_model(
        eval_model.model, model_dir, eval_sess, "eval")

  eval_data_size = len(eval_data)
  eval_data = utils.get_batches(eval_data, args.max_batch)

  eval_total_loss = 0.0
  total_predict_count = 0.0
  for idx, batch in enumerate(eval_data):
    eval_loss, predict_count, batch_size = loaded_eval_model.eval(eval_sess, 
        batch.user, batch.product, batch.rating, batch.review_input, batch.review_output, 
        batch.review_length)
    eval_total_loss += eval_loss * batch_size
    total_predict_count += predict_count

  eval_avg_loss = eval_total_loss / eval_data_size
  eval_ppl = np.exp(eval_total_loss / total_predict_count)
  return eval_avg_loss, eval_ppl

def main(args):
  # main
  save_dir = args.save_dir
  num_train_steps = args.max_step

  train_model = model_helper.build_train_model(args, use_attention=False)
  eval_model = model_helper.build_eval_model(args, use_attention=False)
  infer_model = model_helper.build_infer_model(args, use_attention=False)

  config_proto = utils.get_config_proto()
  train_sess = tf.Session(config=config_proto, graph=train_model.graph)
  eval_sess = tf.Session(config=config_proto, graph=eval_model.graph)
  infer_sess = tf.Session(config=config_proto, graph=infer_model.graph)

  sample_data = utils.load_data(args.infer_dir)

  with train_model.graph.as_default():
    loaded_train_model, global_step = model_helper.create_or_load_model(
        train_model.model, args.save_dir, train_sess, name="train")

  train_sess.run(train_model.iterator.initializer)
  for step in range(num_train_steps):
    start_time = time.time()
    try:
      step_result = loaded_train_model.train(train_sess)
      (_, train_loss, train_rec_loss, train_kl_loss, train_rec_ppl, global_step, \
          predict_count, batch_size) = step_result
      if global_step % args.print_step == 0:
        print "global step %d, loss %.9f, rec_loss %.9f, kl_loss %.9f, ppl: %.2f, time %.2fs" % \
            (global_step, train_loss, train_rec_loss, train_kl_loss, train_rec_ppl, \
              time.time() - start_time)
        start_time = time.time()
      if global_step % args.eval_step == 0:
        loaded_train_model.saver.save(train_sess,
            os.path.join(args.save_dir, "vrae.ckpt"), global_step=global_step)
      #  eval_start_time = time.time()
      #   eval_avg_loss, eval_ppl = run_internal_eval(
      #       args, eval_model, eval_sess, args.save_dir, eval_data_vec)
      #   print "eval loss: %f, eval ppl: %.2f after training of step %d, time %.2fs" % \
      #       (eval_avg_loss, eval_ppl, global_step, time.time() - eval_start_time)
        run_sample_decode(args, infer_model, infer_sess, args.save_dir, sample_data)
        step_start_time = time.time()
      # if args.anneal and global_step > (train_data_size / batch_size) * args.anneal_start:
      #   train_sess.run(train_model.learning_rate_decay_op)
    except tf.errors.OutOfRangeError:
      train_sess.run(train_model.iterator.initializer)
      print "one epoch finish, time %.2fs" % (time.time() - start_time)
      continue

if __name__ == '__main__':
  args = config.get_args()
  main(args)
