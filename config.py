
import argparse

def get_args():
  """
    The argument parser
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--random_seed', type=int, default=1013, help='random seed')

  parser.add_argument('--train_dir', type=str, default='data/ptb.train.txt', help='train data dir')
  parser.add_argument('--eval_dir', type=str, default='data/ptb.dev.txt', help='dev data dir')
  parser.add_argument('--infer_dir', type=str, default='data/ptb.test.txt', help='test data dir')
  parser.add_argument('--vocab_dir', type=str, default='data/vocab.txt', help='vocab dir')
  parser.add_argument('--glove_dir', type=str, default=None, help='pretrain word vector dir')
  parser.add_argument('--save_dir', type=str, default='save', help='save directory')
  parser.add_argument('--batch_size', type=int, default=32, help='train batch size')
  parser.add_argument('--max_batch', type=int, default=1000, help='max batch size at eval')

  parser.add_argument('--hidden_size', type=int, default=512, help='dims of rnn hidden layer')
  parser.add_argument('--latent_size', type=int, default=32, help="dims of latent variable")
  parser.add_argument('--forget_bias', type=float, default=1., help='forget bias of cell')
  parser.add_argument('--num_layers', type=int, default=1, help='layers number of rnn')
  parser.add_argument('--vocab_size', type=int, default=10001, help='vocab size')
  parser.add_argument('--embed_size', type=int, default=512, help='dimension of embed')
  parser.add_argument('--encoder_type', type=str, default="uni", help='encoder type')
  parser.add_argument('--attention_option', type=str, default='bahdanau', help='attention option')
  parser.add_argument('--beam_width', type=int, default=0, help='width of beam search')

  parser.add_argument('--max_epoch', type=int, default=20, help='number of epoch')
  parser.add_argument('--anneal', type=bool, default=True, help='whether to anneal')
  parser.add_argument('--anneal_start', type=int, default=10, help='anneal start epoch')
  parser.add_argument('--max_step', type=int, default=50000, help='max train step')
  parser.add_argument('--init_w', type=float, default=0.08, help='weight init')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
  parser.add_argument('--lr_decay', type=float, default=0.97, help='learning rate decay rate')
  parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
  parser.add_argument('--max_grad_norm', type=float, default=5.0, help='max norm of gradient')
  parser.add_argument('--print_step', type=int, default=100, help='print information step')
  parser.add_argument('--eval_step', type=int, default=100, help='eval model step')
  
  return parser.parse_args()
