import math
import copy
import torch
import torch.nn as nn
from time import time 

from utils.process_data import *
from models.Bi_LSTM import attention, AttnEncoderDecoder



def causal_mask(T):
    """
    Generate a causal mask.
    Arguments:
      T: the length of target sequence
    Returns:
      mask: a T x T tensor, where `mask[i, j]` should be `True` 
      if y_i can attend to y_{j-1} (there's a "-1" since the first 
      token in decoder input is <bos>) and `False` if y_i cannot 
      attend to y_{j-1}
    """
    mask = torch.ones([T, T], dtype=torch.bool)
    mask = torch.triu(mask, diagonal=1)
    mask = torch.logical_not(mask)
    
    return mask.to(device)

class TransformerEncoderDecoder(AttnEncoderDecoder):
  def __init__(self, src_field, tgt_field, hidden_size=64, layers=3, name="TransformerEncoderDecoder"):
    """
    Initializer. Creates network modules and loss function.
    Arguments:
        src_field: src field
        tgt_field: tgt field
        hidden_size: hidden layer size of both encoder and decoder
        layers: number of layers of both encoder and decoder
    """
    super(AttnEncoderDecoder, self).__init__()
    self.src_field = src_field
    self.tgt_field = tgt_field
    self.name = name
    
    # Keep the vocabulary sizes available
    self.V_src = len(src_field.vocab.itos)
    self.V_tgt = len(tgt_field.vocab.itos)
    
    # Get special word ids
    self.padding_id_src = src_field.vocab.stoi[src_field.pad_token]
    self.padding_id_tgt = tgt_field.vocab.stoi[tgt_field.pad_token]
    self.bos_id = tgt_field.vocab.stoi[tgt_field.init_token]
    self.eos_id = tgt_field.vocab.stoi[tgt_field.eos_token]

    # Keep hyper-parameters available
    self.embedding_size = hidden_size
    self.hidden_size = hidden_size
    self.layers = layers

    # Create essential modules
    self.encoder = TransformerEncoder(self.V_src, hidden_size, layers)
    self.decoder = TransformerDecoder(self.V_tgt, hidden_size, layers)

    # Final projection layer
    self.hidden2output = nn.Linear(hidden_size, self.V_tgt)
   
    # Create loss function
    self.loss_function = nn.CrossEntropyLoss(reduction='sum', 
                                             ignore_index=self.padding_id_tgt)

  def forward_encoder(self, src, src_lengths):
    """
    Encodes source words `src`.
    Arguments:
        src: src batch of size (max_src_len, bsz)
        src_lengths: src lengths (bsz)
    Returns:
        memory_bank: a tensor of size (src_len, bsz, hidden_size) 
    """
    # The reason we don't directly pass in src_mask as in `forward_decoder` is to
    # enable us to reuse beam search implemented for RNN-based encoder-decoder
    src_len = src.size(0)
    #TODO - compute `src_mask`
    src_mask = src.ne(self.padding_id_src)
    memory_bank = self.encoder(src, src_mask)
    return memory_bank, None

  def forward_decoder(self, tgt_in, memory_bank, src_mask):
    """
    Decodes based on memory bank, and ground truth target words.
    Arguments:
        tgt_in: a tensor of size (tgt_len, bsz)
        memory_bank: a tensor of size (src_len, bsz, hidden_size), encoder outputs 
                     at every position
    Returns:
        Logits of size (tgt_len, bsz, V_tgt) (before the softmax operation)
    """
    tgt_len = tgt_in.size(0)
    bsz = tgt_in.size(1)
    #TODO - compute `src_mask` and `tgt_mask`, note that the src_mask here has a different
    #       shape as the `src_mask` passed in, as the attention function needs
    #       a mask of the same size as the attention matrix
    src_mask = src_mask.transpose(0, 1)
    src_mask = torch.unsqueeze(src_mask, dim = 1)
    src_mask = src_mask.repeat((1, tgt_len, 1))
    
    tgt_mask = causal_mask(tgt_len)
    tgt_mask = torch.unsqueeze(tgt_mask, dim=0)
    tgt_mask = tgt_mask.repeat((bsz, 1, 1))
    
    outputs = self.decoder(tgt_in, memory_bank, src_mask, tgt_mask)
    logits = self.hidden2output(outputs)
    return logits

  def forward(self, src, src_lengths, tgt_in):
    """
    Performs forward computation, returns logits.
    Arguments:
        src: src batch of size (max_src_len, bsz)
        src_lengths: src lengths of size (bsz)
        tgt_in:  a tensor of size (tgt_len, bsz)
    """
    src_mask = src.ne(self.padding_id_src) # max_src_len, bsz
    # Forward encoder
    memory_bank, _ = self.forward_encoder(src, src_lengths)
    # Forward decoder
    logits = self.forward_decoder(tgt_in, memory_bank, src_mask)
    return logits

  def forward_decoder_incrementally(self, prev_decoder_states, tgt_in_onestep, 
                                    memory_bank, src_mask, normalize=True):
    """
    Forward the decoder at `decoder_state` for a single step with token `tgt_in_onestep`.
    This function will be used in beam search. Note that the implementation here is
    very inefficient, since we do not cache any decoder state, but instead we only
    cache previously generated tokens in `prev_decoder_states`, and do a fresh
    `forward_decoder`.
    Arguments:
        prev_decoder_states: previous tgt words. None for the first step.
        tgt_in_onestep: a tensor of size (bsz), tokens at one step
        memory_bank: a tensor of size (src_len, bsz, hidden_size), src hidden states 
                     at every position
        src_mask: a tensor of size (src_len, bsz): a boolean tensor, `False` where
                  src is padding.
        normalize: use log_softmax to normalize or not. Beam search needs to normalize,
                   while `forward_decoder` does not
    Returns:
        logits: Log probabilities for `tgt_in_token` of size (bsz, V_tgt)
        decoder_states: we use tgt words up to now as states, a tensor of size (len, bsz)
        None: to keep output format the same as AttnEncoderDecoder, such that we can
              reuse beam search code
        
    """
    prev_tgt_in = prev_decoder_states # tgt_len, bsz
    src_len = memory_bank.size(0)
    bsz = memory_bank.size(1)
    tgt_in_onestep = tgt_in_onestep.view(1, -1) # 1, bsz
    if prev_tgt_in is not None:
      tgt_in = torch.cat((prev_tgt_in, tgt_in_onestep), 0) # tgt_len+1, bsz
    else:
      tgt_in = tgt_in_onestep
    tgt_len = tgt_in.size(1)
    
    logits = self.forward_decoder(tgt_in, memory_bank, src_mask)
    logits = logits[-1]
    if normalize:
      logits = torch.log_softmax(logits, dim=-1)
    decoder_states = tgt_in
    return logits, decoder_states, None


class TransformerEncoder(nn.Module):
  r"""TransformerEncoder is an embedding layer and a stack of N encoder layers.
  Arguments:
      hidden_size: hidden size.
      layers: the number of encoder layers.
  """

  def __init__(self, vocab_size, hidden_size, layers):
    super(TransformerEncoder, self).__init__()
    self.embed = PositionalEmbedding(vocab_size, hidden_size)
    encoder_layer = TransformerEncoderLayer(hidden_size)
    self.layers = _get_clones(encoder_layer, layers)
    self.norm = nn.LayerNorm(hidden_size)

  def forward(self, src, src_mask):
    r"""Pass the input through the word embedding layer, followed by
    the encoder layers in turn.
    Arguments:
        src: src batch of size (max_src_len, bsz)
        src_mask: the mask for the src sequence, (max_src_len, bsz)
    Returns:
        a tensor of size (max_src_len, bsz, hidden_size)
    """
    output = self.embed(src)
    for mod in self.layers:
      output = mod(output, src_mask=src_mask)
    output = self.norm(output)
    return output


class TransformerEncoderLayer(nn.Module):
  r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
  Arguments:
      hidden_size: hidden size.
  """

  def __init__(self, hidden_size):
    super(TransformerEncoderLayer, self).__init__()
    self.hidden_size = hidden_size
    fwd_hidden_size = hidden_size * 4
    
    # Create modules
    self.linear1 = nn.Linear(hidden_size, fwd_hidden_size)
    self.linear2 = nn.Linear(fwd_hidden_size, hidden_size)
    self.norm1 = nn.LayerNorm(hidden_size)
    self.norm2 = nn.LayerNorm(hidden_size)
    self.activation = nn.ReLU()
    # Attention related
    self.q_proj = nn.Linear(hidden_size, hidden_size)
    self.k_proj = nn.Linear(hidden_size, hidden_size)
    self.v_proj = nn.Linear(hidden_size, hidden_size)
    self.context_proj = nn.Linear(hidden_size, hidden_size)


  def forward(self, src, src_mask):
    r"""Pass the input through the encoder layer.
    Arguments:
        src: an input tensor of size (max_src_len, bsz, hidden_size).
        src_mask: attention mask of size (max_src_len, bsz), it's `False`
        where the corresponding source word is padding.
    Returns:
        a tensor of size (max_src_len, bsz, hidden_size).
    """
    # Attend
    q = self.q_proj(src) / math.sqrt(self.hidden_size) # a trick needed to make transformer work
    k = self.k_proj(src)
    v = self.v_proj(src)
    #TODO - compute `context`
    
    #reshape src 
    src_mask = torch.transpose(src_mask, 0, 1)
    src_mask = src_mask.reshape(src_mask.size(0), 1, src_mask.size(1))
    src_mask = src_mask.repeat((1, src_mask.size(2), 1))
    scores, context = attention(q, k, v, src_mask)
    
    src2 = self.context_proj(context)
    # Residual connection
    src = src + src2
    src = self.norm1(src)
    # Feedforward for each position
    src2 = self.linear2(self.activation(self.linear1(src)))
    src = src + src2
    src = self.norm2(src)
    return src


class TransformerDecoder(nn.Module):
  r"""TransformerDecoder is an embedding layer and a stack of N decoder layers.
  Arguments:
      hidden_size: hidden size.
      layers: the number of sub-encoder-layers in the encoder.
  """
  def __init__(self, vocab_size, hidden_size, layers):
    super(TransformerDecoder, self).__init__()
    self.embed = PositionalEmbedding(vocab_size, hidden_size)
    decoder_layer = TransformerDecoderLayer(hidden_size)
    self.layers = _get_clones(decoder_layer, layers)
    self.norm = nn.LayerNorm(hidden_size)

  def forward(self, tgt_in, memory, src_mask, tgt_mask):
    r"""Pass the inputs (and mask) through the word embedding layer, followed by
    the decoder layer in turn.
    Arguments:
        tgt_in: tgt batch of size (max_tgt_len, bsz)
        memory: the outputs of the encoder (max_src_len, bsz, hidden_size)
        src_mask: attention mask of size (max_src_len, bsz), it's `False`
                  where the corresponding source word is padding.
    Returns:
        a tensor of size (max_tgt_len, bsz, hidden_size)
    """
    output = self.embed(tgt_in)
    for mod in self.layers:
      output = mod(output, memory, src_mask=src_mask, tgt_mask=tgt_mask)

    output = self.norm(output)
    return output


class TransformerDecoderLayer(nn.Module):
  r"""TransformerDecoderLayer is made up of self-attn, cross-attn, and 
  feedforward network.
  Arguments:
      hidden_size: hidden size.
  """

  def __init__(self, hidden_size):
    super(TransformerDecoderLayer, self).__init__()
    self.hidden_size = hidden_size
    fwd_hidden_size = hidden_size * 4
    
    # Create modules
    self.linear1 = nn.Linear(hidden_size, fwd_hidden_size)
    self.linear2 = nn.Linear(fwd_hidden_size, hidden_size)
    
    self.activation = nn.ReLU()

    self.norm1 = nn.LayerNorm(hidden_size)
    self.norm2 = nn.LayerNorm(hidden_size)
    self.norm3 = nn.LayerNorm(hidden_size)
    
    # Attention related
    self.q_proj_self = nn.Linear(hidden_size, hidden_size)
    self.k_proj_self = nn.Linear(hidden_size, hidden_size)
    self.v_proj_self = nn.Linear(hidden_size, hidden_size)
    self.context_proj_self = nn.Linear(hidden_size, hidden_size)
    
    self.q_proj_cross = nn.Linear(hidden_size, hidden_size)
    self.k_proj_cross = nn.Linear(hidden_size, hidden_size)
    self.v_proj_cross = nn.Linear(hidden_size, hidden_size)
    self.context_proj_cross = nn.Linear(hidden_size, hidden_size)

  def forward(self, tgt, memory, src_mask, tgt_mask):
    r"""Pass the inputs (and mask) through the decoder layer.
    Arguments:
        tgt: an input tensor of size (max_tgt_len, bsz, hidden_size).
        memory: encoder outputs of size (max_src_len, bsz, hidden_size).
        src_mask: attention mask of size (bsz, max_tgt_len, max_src_len), 
                  it's `False` where the cross-attention is disallowed.
        tgt_mask: attention mask of size (bsz, max_tgt_len, max_tgt_len),
                  it's `False` where the self-attention is disallowed.
    Returns:
        a tensor of size (max_tgt_len, bsz, hidden_size)
    """
    # Self attention (decoder-side)
    q = self.q_proj_self(tgt) / math.sqrt(self.hidden_size)
    k = self.k_proj_self(tgt)
    v = self.v_proj_self(tgt)
    #TODO - compute `context`
    scores, context = attention(q, k, v, mask=tgt_mask)
    tgt2 = self.context_proj_self(context)
    tgt = tgt + tgt2
    tgt = self.norm1(tgt)
    # Cross attention (decoder attends to encoder)
    q = self.q_proj_cross(tgt) / math.sqrt(self.hidden_size)
    k = self.k_proj_cross(memory)
    v = self.v_proj_cross(memory)
    #TODO - compute `context`
    scores, context = attention(q, k, v, mask=src_mask)
    tgt2 = self.context_proj_cross(context)
    tgt = tgt + tgt2
    tgt = self.norm2(tgt)
    tgt2 = self.linear2(self.activation(self.linear1(tgt)))
    tgt = tgt + tgt2
    tgt = self.norm3(tgt)
    return tgt

class PositionalEmbedding(nn.Module):
  """"Embeds a word both by its word id and by its position in the sentence."""
  def __init__(self, vocab_size, embedding_size, max_len=1024):
    super(PositionalEmbedding, self).__init__()
    self.embedding_size = embedding_size

    self.embed = nn.Embedding(vocab_size, embedding_size)
    pe = torch.zeros(max_len, embedding_size)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_size, 2) *
                         -(math.log(10000.0) / embedding_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(1) # max_len, 1, embedding_size
    self.register_buffer('pe', pe)

  def forward(self, batch):
    x = self.embed(batch) * math.sqrt(self.embedding_size) # type embedding
    # Add positional encoding to type embedding
    x = x + self.pe[:x.size(0)].detach()
    return x


def _get_clones(module, N):
  """Copies a module `N` times"""
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def main(): 
  EPOCHS = 2 # epochs, we highly recommend starting with a smaller number like 1
  LEARNING_RATE = 2e-3 # learning rate

  # Instantiate and train classifier
  model_transformer = TransformerEncoderDecoder(SRC, TGT,
    hidden_size    = 64,
    layers         = 3,
  ).to(device)

  start_time = time()
  model_transformer.train_all(train_iter, val_iter, epochs=EPOCHS, learning_rate=LEARNING_RATE)
  model_transformer.load_state_dict(model_transformer.best_model)
  print("TransformerEncoderDecoder took {} seconds to train.".format(time() - start_time))

if __name__ == '__main__':
  main()