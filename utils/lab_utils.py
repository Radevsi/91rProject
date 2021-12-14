import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = "/../data"

def read_sentence_file(name):
  filename = os.path.join(DATA_DIR, name)

  sentences_list = []
  with open(filename, "r") as f:
    for line in f:
      sentences_list.append(line.strip().split())
  return sentences_list

def read_vocab_file(name):
  filename = os.path.join(DATA_DIR, name)

  with open(filename, "r") as f:
    # Need to add pad bc vocab doesn't have it...
    return ['<pad>'] + [line.strip() for line in f]

MAX_SENT_LENGTH = 48
MAX_SENT_LENGTH_PLUS_SOS_EOS = 50

# We only keep sentences that do not exceed 48 words, so that later when we
# add <s> and </s> to a sentence it still won't exceed 50 words.
def filter_data(src_sentences_list, trg_sentences_list, max_len):
  new_src_sentences_list, new_trg_sentences_list = [], []
  for src_sent, trg_sent in zip(src_sentences_list, trg_sentences_list):
    if (len(src_sent) <= max_len and len(trg_sent) <= max_len
        and len(src_sent) > 0 and len(trg_sent)) > 0:
      new_src_sentences_list.append(src_sent)
      new_trg_sentences_list.append(trg_sent)
  return new_src_sentences_list, new_trg_sentences_list

# Show some data stats
def show_some_data_stats(train_src_sentences_list, val_src_sentences_list, 
                     test_src_sentences_list, train_trg_sentences_list,
                     src_vocab_set, trg_vocab_set):
	print("Number of training (src, trg) sentence pairs: %d" %
	      len(train_src_sentences_list))
	print("Number of validation (src, trg) sentence pairs: %d" %
	      len(val_src_sentences_list))
	print("Number of testing (src, trg) sentence pairs: %d" %
	      len(test_src_sentences_list))
	src_vocab_set = ['<pad>'] + src_vocab_set
	trg_vocab_set = ['<pad>'] + trg_vocab_set
	print("Size of en vocab set (including '<pad>', '<unk>', '<s>', '</s>'): %d" %
	      len(src_vocab_set))
	print("Size of vi vocab set (including '<pad>', '<unk>', '<s>', '</s>'): %d" %
	      len(trg_vocab_set))

	length = [len(sent) for sent in train_src_sentences_list]
	print('Training sentence avg. length: %d ' % np.mean(length))
	print('Training sentence length at 95-percentile: %d' %
	      np.percentile(length, 95))
	print('Training sentence length distribution '
	      '(x-axis is length range and y-axis is count):\n')
	plt.hist(length, bins=5)
	plt.show()

	print('Example Vietnamese input: ' + str(train_src_sentences_list[0]))
	print('Its target English output: ' + str(train_trg_sentences_list[0]))

###################################################################################
###################################################################################

def plot_perplexity(perplexities):
  """plot perplexities"""
  plt.title("Perplexity per Epoch")
  plt.xlabel("Epoch")
  plt.ylabel("Perplexity")
  plt.plot(perplexities)

###################################################################################
###################################################################################

import torch 
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda"   # use gpu whenever you can!

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def lookup_words(x, vocab):
  return [vocab[i] for i in x]

def print_examples(model, src_vocab_set, trg_vocab_set, data_loader, decoder, 
                   with_attention=False, n=3, EOS_INDEX=3, max_len=MAX_SENT_LENGTH_PLUS_SOS_EOS):
  """Prints `n` examples. Assumes batch size of 1."""

  model.eval()

  for i, (src_ids, src_lengths, trg_ids, _) in enumerate(data_loader):
    if not with_attention:
      result = decoder(model, src_ids.to(device), src_lengths.to(device),
                             max_len=max_len)
    else:
      result, _ = decoder(model, src_ids.to(device),
                                          src_lengths.to(device),
                                          max_len=max_len)

    # remove <s>
    src_ids = src_ids[0, 1:]
    trg_ids = trg_ids[0, 1:]
    # remove </s> and <pad>
    src_ids = src_ids[:np.where(src_ids == EOS_INDEX)[0][0]]
    trg_ids = trg_ids[:np.where(trg_ids == EOS_INDEX)[0][0]]
  
    print("Example #%d" % (i + 1))
    print("Src : ", " ".join(lookup_words(src_ids, vocab=src_vocab_set)))
    print("Trg : ", " ".join(lookup_words(trg_ids, vocab=trg_vocab_set)))
    print("Pred: ", " ".join(lookup_words(result, vocab=trg_vocab_set)))
    print()

    if i == n - 1:
      break
