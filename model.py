
import tensorflow as tf
import numpy as np

class BaseModel(object):
  def __init__(self, 
             args, 
             mode,
             iterator,
             vocab_table,
             reverse_vocab_table=None,
             scope=None,
             name=None):
    self.vocab_size = args.vocab_size
    self.embed_size = args.embed_size
    self.num_layers = args.num_layers
    self.hidden_size = args.hidden_size
    self.latent_size = args.latent_size
    self.forget_bias = args.forget_bias
    self.dropout = args.dropout
    self.encoder_type = args.encoder_type #bi-lstm or not
    self.beam_width = args.beam_width
    self.max_grad_norm = args.max_grad_norm

    self.mode = mode
    self.vocab_table = vocab_table
    self.scope = scope
    self.iterator = iterator
    self.initializer = tf.random_uniform_initializer(-args.init_w, args.init_w)
    self.learning_rate = tf.Variable(float(args.learning_rate), 
        trainable=False, name="learning_rate")
    self.learning_rate_decay_op = self.learning_rate.assign(
        tf.multiply(self.learning_rate, args.lr_decay))
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

    with tf.variable_scope("sequence_decoder"):
      with tf.variable_scope("output_projection"):
        self.output_layer = tf.layers.Dense(
            self.vocab_size, use_bias=False, name="output_projection")

    self.batch_size = tf.size(self.iterator.source_sequence_length)

    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      ## Count the number of predicted words for compute ppl.
      self.predict_count = tf.reduce_sum(self.iterator.target_sequence_length)

    self.global_step = tf.Variable(0, trainable=False)

  def build_graph(self):
    raise NotImplementedError("Implement how to build graph")

  def build_encoder(self):
    iterator = self.iterator
    source = iterator.source
    source_sequence_length = iterator.source_sequence_length
    with tf.variable_scope("encoder") as scope:
      encoder_embed = self._build_embedding(self.vocab_size, 
          self.embed_size, "encoder_embedding")
      embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_size)], 
          dtype=tf.float32, shape=[self.vocab_size, 1])
      encoder_embed = encoder_embed * embedding_mask
      encoder_embed_inp = tf.nn.embedding_lookup(encoder_embed, source)

      if self.encoder_type == "uni":
        encoder_cell = self._build_encoder_cell(self.hidden_size, 
            self.forget_bias, self.num_layers, self.mode, self.initializer, self.dropout)
        encoder_output, encoder_state = tf.nn.dynamic_rnn(
            cell=encoder_cell, 
            inputs=encoder_embed_inp, 
            dtype=tf.float32,
            sequence_length=source_sequence_length,
            swap_memory=True)

      elif self.encoder_type == "bi":
        num_bi_layers = self.num_layers / 2
        fw_cell = self._build_encoder_cell(self.hidden_size, 
            self.forget_bias, num_bi_layers, self.mode, self.initializer, self.dropout)
        bw_cell = self._build_encoder_cell(self.hidden_size, 
            self.forget_bias, num_bi_layers, self.mode, self.initializer, self.dropout)
        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell, 
            cell_bw=bw_cell, 
            inputs=encoder_embed_inp,
            dtype=tf.float32, 
            sequence_length=source_sequence_length,
            swap_memory=True)
        encoder_output = tf.concat(bi_outputs, -1)

        if num_bi_layers == 1:
          encoder_state = bi_state
        else:
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_state[0][layer_id])   #forward
            encoder_state.append(bi_state[1][layer_id])   #backward
          encoder_state = tuple(encoder_state)

      else:
        raise ValueError("Unknown encoder_type %s" % self.encoder_type)

    return encoder_output, encoder_state

  def _single_cell(self, num_units, forget_bias, mode, initializer, dropout):
    """ single cell """
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
    cell = tf.contrib.rnn.LSTMCell(
        num_units=num_units,
        forget_bias=forget_bias,
        initializer=initializer,
        state_is_tuple=True)
    if dropout > 0.0:
      print"use dropout, dropout rate: %.2f" % dropout
      cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=(1.0 - dropout))
    return cell

  def _build_encoder_cell(self, num_units, forget_bias, num_layers, mode,
                          initializer, dropout=0.0):

    cell_list = []
    for i in range(num_layers):
      cell = self._single_cell(num_units, forget_bias, mode, initializer, dropout)
      cell_list.append(cell)

    if num_layers == 1:
      return cell_list[0]
    else:
      return tf.contrib.rnn.MultiRNNCell(cell_list)

  def _get_infer_maximum_iterations(self, source_sequence_length):
    decoding_length_factor = 2.0
    max_sequence_length = tf.reduce_max(source_sequence_length)
    maximum_iterations = tf.to_int32(tf.round(
        tf.to_float(max_sequence_length) * decoding_length_factor))
    return maximum_iterations

  def get_max_time(self, tensor):
    time_axis = 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

  def _weight_variable(self, shape, name, initializer=None):
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

  def _build_embedding(self, vocab_size, embed_size, name):
    initializer = self.initializer
    with tf.variable_scope("embedding") as scope:
      embedding = self._weight_variable([vocab_size, embed_size], 
                                        name=name, 
                                        initializer=initializer)
    return embedding

class RAE(BaseModel):
  def __init__(self, 
             args, 
             mode,
             iterator,
             vocab_table,
             reverse_vocab_table=None,
             scope=None,
             name=None):

    super(RAE, self).__init__(
        args=args,
        mode=mode,
        iterator=iterator,
        vocab_table=vocab_table,
        reverse_vocab_table=reverse_vocab_table,
        scope=scope,
        name=name)

    res = self.build_graph()
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_loss = res[1]
      self.train_ppl = res[2]
    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      self.eval_loss = res[1]
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_logits, _, _, self.final_state, self.sample_id = res
      self.sample_words = reverse_vocab_table.lookup(tf.to_int64(self.sample_id))

    params = tf.trainable_variables()
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      gradients = tf.gradients(
          self.train_loss, params)
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
      self.update = self.optimizer.apply_gradients(
          zip(clipped_gradients, params), global_step=self.global_step)

    self.tvars = tf.trainable_variables()
    self.saver = tf.train.Saver(tf.global_variables())

  def build_graph(self):
    """ model graph """
    with tf.variable_scope("seq2seq"):
      encoder_output, encoder_state = self.build_encoder()
      logits, sample_id, final_state, = self.build_decoder(encoder_output, encoder_state)

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        loss, ppl = self.build_loss(logits)
      else:
        loss, ppl = None, None
    return logits, loss, ppl, final_state, sample_id

  def build_decoder(self, encoder_output, encoder_state):
    iterator = self.iterator
    source_sequence_length = iterator.source_sequence_length

    tgt_sos_id = tf.cast(self.vocab_table.lookup(tf.constant("<s>")), tf.int32)
    tgt_eos_id = tf.cast(self.vocab_table.lookup(tf.constant("</s>")), tf.int32)

    with tf.variable_scope("decoder") as decoder_scope:
      decoder_cell, decoder_initial_state = self._build_decoder_cell(
          self.hidden_size, 
          self.forget_bias, 
          self.num_layers, 
          encoder_output,
          encoder_state,
          self.mode,
          self.initializer,
          self.dropout)

      decoder_embed = self._build_embedding(self.vocab_size, 
          self.embed_size, "decoder_embedding")
      embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_size)], 
          dtype=tf.float32, shape=[self.vocab_size, 1])
      decoder_embed = decoder_embed * embedding_mask

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        target_input = iterator.target_input
        target_sequence_length = iterator.target_sequence_length

        decoder_embed_inp = tf.nn.embedding_lookup(decoder_embed, target_input)
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=decoder_embed_inp, 
            sequence_length=target_sequence_length, 
            name="decoder_helper")

        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell, 
            helper=helper, 
            initial_state=decoder_initial_state)

        output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder, swap_memory=True, scope=decoder_scope)

        sample_id = output.sample_id
        logits = self.output_layer(output.rnn_output)

      else:
        start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        end_token = tgt_eos_id

        maximum_iterations = self._get_infer_maximum_iterations(source_sequence_length)
        if self.beam_width > 0:
          my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=decoder_cell, 
              embedding=decoder_embed, 
              start_tokens=start_tokens, 
              end_token=end_token, 
              initial_state=decoder_initial_state, 
              beam_width=self.beam_width,
              output_layer=self.output_layer)
        else:
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
              embedding=decoder_embed, 
              start_tokens=start_tokens, 
              end_token=end_token)
          my_decoder = tf.contrib.seq2seq.BasicDecoder(
              cell=decoder_cell,
              helper=helper,
              initial_state=decoder_initial_state,
              output_layer=self.output_layer)

        output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder, 
            maximum_iterations=maximum_iterations, 
            swap_memory=True, 
            scope=decoder_scope)

        if self.beam_width > 0:
          logits = tf.no_op()
          sample_id = output.predicted_ids
        else:
          logits = output.rnn_output
          sample_id = output.sample_id

    return logits, sample_id, final_state

  def build_loss(self, logits, mu, logvar):
    """ compute loss """
    iterator = self.iterator
    max_len = self.get_max_time(iterator.target_input)
    weight = tf.sequence_mask(iterator.target_sequence_length, max_len, dtype=logits.dtype)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=iterator.target_output, logits=logits)
    loss = tf.reduce_sum(rec_loss * weight, 1)
    avg_loss = tf.reduce_mean(rec_loss)
    ppl = tf.exp(tf.reduce_sum(loss) / tf.to_float(tf.reduce_sum(
        iterator.target_sequence_length)))
    return avg_loss, ppl

  def train(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    return sess.run([self.update, 
                     self.train_loss,
                     self.train_ppl,
                     self.global_step,
                     self.predict_count,
                     self.batch_size])

  def eval(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.EVAL
    return sess.run([self.eval_loss,
                     self.predict_count,
                     self.batch_size])

  def infer(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    _, sample_id, sample_words = sess.run(
        [self.infer_logits, self.sample_id, self.sample_words])
    return sample_id[0], sample_words[0]

  def _build_decoder_cell(self, num_units, forget_bias, num_layers,
                          encoder_output, encoder_state, mode,
                          initializer, dropout=0.0):

    cell_list = []
    for i in range(num_layers):
      cell = self._single_cell(num_units, forget_bias, mode, initializer, dropout)
      cell_list.append(cell)

    if num_layers == 1:
      cell = cell_list[0]
    else:
      cell = tf.contrib.rnn.MultiRNNCell(cell_list)

    if mode == tf.contrib.learn.ModeKeys.INFER and self.beam_width > 0:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(decoder_initial_state, self.beam_width)
    else:
      decoder_initial_state = decoder_initial_state
    return cell, decoder_initial_state

class VRAE(BaseModel):
  def __init__(self, 
               args, 
               mode,
               iterator,
               vocab_table,
               reverse_vocab_table=None,
               scope=None,
               name=None):
    super(VRAE, self).__init__(
        args=args,
        mode=mode,
        iterator=iterator,
        vocab_table=vocab_table,
        reverse_vocab_table=reverse_vocab_table,
        scope=scope,
        name=name)

    res = self.build_graph()
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_rec_loss = res[1]
      self.train_rec_ppl = res[2]
      self.train_kl_loss = res[3]
      self.train_loss = res[4]
    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      self.eval_rec_loss = res[1]
      self.eval_kl_loss = res[3]
      self.eval_loss = res[4]
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_logits, _, _, _, _, self.final_state, self.sample_id = res
      self.sample_words = reverse_vocab_table.lookup(tf.to_int64(self.sample_id))

    params = tf.trainable_variables()
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      gradients = tf.gradients(
          self.train_loss, params)
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
      self.update = self.optimizer.apply_gradients(
          zip(clipped_gradients, params), global_step=self.global_step)

    self.tvars = tf.trainable_variables()
    self.saver = tf.train.Saver(tf.global_variables())
    
  def build_graph(self):
    """ model graph """
    with tf.variable_scope("seq2seq"):
      encoder_output, encoder_state = self.build_encoder()
      encoder_state = tf.concat(encoder_state, 1)
      mu, logvar = self.build_latent(encoder_state)
      z = self.sample_z(mu, logvar)
      logits, sample_id, final_state, = self.build_decoder(z)

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        rec_loss, rec_ppl, kl_loss, loss = self.build_loss(logits, mu, logvar)
      else:
        rec_loss, rec_ppl, kl_loss, loss = None, None, None, None
    return logits, rec_loss, rec_ppl, kl_loss, loss, final_state, sample_id

  def build_latent(self, encoder_state):
    with tf.variable_scope("latent") as scope:
      mulogvar = tf.layers.dense(encoder_state, self.latent_size * 2, activation=None)
      mu, logvar = tf.split(mulogvar, 2, axis=1)
    return mu, logvar

  def sample_z(self, mu, log_sigma_sq):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_sigma_sq / 2.) * eps

  def build_decoder(self, z):
    iterator = self.iterator
    source_sequence_length = iterator.source_sequence_length

    tgt_sos_id = tf.cast(self.vocab_table.lookup(tf.constant("<s>")), tf.int32)
    tgt_eos_id = tf.cast(self.vocab_table.lookup(tf.constant("</s>")), tf.int32)

    z = tf.layers.dense(z, self.hidden_size * 2 * self.num_layers)
    z = tf.reshape(z, [-1, self.num_layers, 2, self.hidden_size])
    with tf.variable_scope("decoder") as decoder_scope:
      decoder_cell, decoder_initial_state = self._build_decoder_cell(
          self.hidden_size, 
          self.forget_bias, 
          self.num_layers, 
          z,
          self.mode,
          self.initializer,
          self.dropout)

      decoder_embed = self._build_embedding(self.vocab_size, 
          self.embed_size, "decoder_embedding")
      embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_size)], 
          dtype=tf.float32, shape=[self.vocab_size, 1])
      decoder_embed = decoder_embed * embedding_mask

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        target_input = iterator.target_input
        target_sequence_length = iterator.target_sequence_length

        decoder_embed_inp = tf.nn.embedding_lookup(decoder_embed, target_input)
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=decoder_embed_inp, 
            sequence_length=target_sequence_length, 
            name="decoder_helper")

        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell, 
            helper=helper, 
            initial_state=decoder_initial_state)

        output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder, swap_memory=True, scope=decoder_scope)

        sample_id = output.sample_id
        logits = self.output_layer(output.rnn_output)

      else:
        start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        end_token = tgt_eos_id

        maximum_iterations = self._get_infer_maximum_iterations(source_sequence_length)
        if self.beam_width > 0:
          my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=decoder_cell, 
              embedding=decoder_embed, 
              start_tokens=start_tokens, 
              end_token=end_token, 
              initial_state=decoder_initial_state, 
              beam_width=self.beam_width,
              output_layer=self.output_layer)
        else:
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
              embedding=decoder_embed, 
              start_tokens=start_tokens, 
              end_token=end_token)
          my_decoder = tf.contrib.seq2seq.BasicDecoder(
              cell=decoder_cell,
              helper=helper,
              initial_state=decoder_initial_state,
              output_layer=self.output_layer)

        output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder, 
            maximum_iterations=maximum_iterations, 
            swap_memory=True, 
            scope=decoder_scope)

        if self.beam_width > 0:
          logits = tf.no_op()
          sample_id = output.predicted_ids
        else:
          logits = output.rnn_output
          sample_id = output.sample_id

    return logits, sample_id, final_state

  def build_loss(self, logits, mu, logvar):
    """ compute loss """
    iterator = self.iterator
    max_len = self.get_max_time(iterator.target_input)
    weight = tf.sequence_mask(iterator.target_sequence_length, max_len, dtype=logits.dtype)
    rec_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=iterator.target_output, logits=logits)
    rec_loss = tf.reduce_sum(rec_loss * weight, 1)
    avg_rec_loss = tf.reduce_mean(rec_loss)
    rec_ppl = tf.exp(tf.reduce_sum(rec_loss) / tf.to_float(tf.reduce_sum(
        iterator.target_sequence_length)))
    kl_loss = -0.5 * tf.reduce_sum(1. + logvar - mu ** 2 - tf.exp(logvar), 1)
    avg_kl_loss = tf.reduce_mean(kl_loss)
    avg_loss = tf.reduce_mean(rec_loss + kl_loss)
    return avg_rec_loss, rec_ppl, avg_kl_loss, avg_loss

  def train(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    return sess.run([self.update, 
                     self.train_loss,
                     self.train_rec_loss,
                     self.train_kl_loss,
                     self.train_rec_ppl,
                     self.global_step,
                     self.predict_count,
                     self.batch_size])

  def eval(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.EVAL
    return sess.run([self.eval_loss,
                     self.eval_rec_loss,
                     self.eval_kl_loss,
                     self.predict_count,
                     self.batch_size])

  def infer(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    _, sample_id, sample_words = sess.run(
        [self.infer_logits, self.sample_id, self.sample_words])
    return sample_id[0], sample_words[0]

  def _build_decoder_cell(self, num_units, forget_bias, num_layers, z, mode,
                          initializer, dropout=0.0):

    cell_list = []
    for i in range(num_layers):
      cell = self._single_cell(num_units, forget_bias, mode, initializer, dropout)
      cell_list.append(cell)

    if num_layers == 1:
      cell = cell_list[0]
    else:
      cell = tf.contrib.rnn.MultiRNNCell(cell_list)

    z = tf.transpose(z, [1, 2, 0, 3])
    layer_unpacked = tf.unstack(z, axis=0)
    lstm_state_tuple = tf.contrib.rnn.LSTMStateTuple(layer_unpacked[0][0], layer_unpacked[0][1])
    # lstm_state_tuple = tuple(
    #   [tf.contrib.rnn.LSTMStateTuple(layer_unpacked[layer][0], layer_unpacked[layer][1]) 
    #   for layer in range(self.num_layers)])

    if mode == tf.contrib.learn.ModeKeys.INFER and self.beam_width > 0:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(lstm_state_tuple, self.beam_width)
    else:
      decoder_initial_state = lstm_state_tuple
    return cell, decoder_initial_state

