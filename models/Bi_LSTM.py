import math
import copy

import torch
import torch.nn as nn

from tqdm import tqdm
from time import time 

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from utils.process_data import *


def attention(batched_Q, batched_K, batched_V, mask=None):
    """
    Performs the attention operation and returns the attention matrix
    `batched_A` and the context matrix `batched_C` using queries 
    `batched_Q`, keys `batched_K`, and values `batched_V`.
    Arguments:
      batched_Q: (q_len, bsz, D)
      batched_K: (k_len, bsz, D)
      batched_V: (k_len, bsz, D)
      mask: (bsz, q_len, k_len). An optional boolean mask *disallowing* 
            attentions where the mask value is *`False`*.
    Returns:
      batched_A: the normalized attention scores (bsz, q_len, k_ken)
      batched_C: a tensor of size (q_len, bsz, D).
    """
    # Check sizes
    D = batched_Q.size(-1)
    bsz = batched_Q.size(1)
    q_len = batched_Q.size(0)
    k_len = batched_K.size(0)
    assert batched_K.size(-1) == D and batched_V.size(-1) == D
    assert batched_K.size(1) == bsz and batched_V.size(1) == bsz
    assert batched_V.size(0) == k_len
    if mask is not None:
        assert mask.size() == torch.Size([bsz, q_len, k_len])

    #reshape to (bsz, q_len, D)
    batched_Q = batched_Q.transpose(0,1)
    
    #reshape to (bsz, D, k_len)
    batched_K = batched_K.transpose(0, 1) #(bsz, k_len, D)
    batched_K = batched_K.transpose(1, 2) 
    
    scores = torch.bmm(batched_Q, batched_K)      
    neginf = -math.inf
    
    if mask is not None:
        scores = scores.masked_fill_(mask==False, neginf)
        
    # size (bsz, q_len, k_ken)
    batched_A = torch.softmax(scores, dim=-1)
    
    # size of V originally(k_len, bsz, D)
    #reshape to (bsz, k_len, D)
    batched_V = batched_V.transpose(0, 1) #(bsz, k_len, D)
    
    batched_C = torch.bmm(batched_A, batched_V)
    
    # reshape batched_c t0 proper return type (q_len, bsz, D)
    batched_C = batched_C.transpose(0, 1) #(bsz, q_len, D) -> (q_len, bsz, D)

    # Verify that things sum up to one properly.
    assert torch.all(torch.isclose(batched_A.sum(-1), 
                                 torch.ones(bsz, q_len).to(device)))

    return batched_A, batched_C


class AttnEncoderDecoder(nn.Module):
  def __init__(self, src_field, tgt_field, hidden_size=64, layers=3, name="AttnEncoderDecoder"):
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
    print(self.V_src, self.V_tgt)
    
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
    self.word_embeddings_src = nn.Embedding(self.V_src, self.embedding_size)
    self.word_embeddings_tgt = nn.Embedding(self.V_tgt, self.embedding_size)

    # RNN cells
    self.encoder_rnn = nn.LSTM(
      input_size    = self.embedding_size,
      hidden_size   = hidden_size // 2, # to match decoder hidden size
      num_layers    = layers,
      bidirectional = True              # bidirectional encoder
    )
    self.decoder_rnn = nn.LSTM(
      input_size    = self.embedding_size,
      hidden_size   = hidden_size,
      num_layers    = layers,
      bidirectional = False             # unidirectional decoder
    )

    # Final projection layer
    self.hidden2output = nn.Linear(2*hidden_size, self.V_tgt) # project the concatenation to logits
   
    # Create loss function
    self.loss_function = nn.CrossEntropyLoss(reduction='sum', 
                                             ignore_index=self.padding_id_tgt)

  def matrix_reshape(self, m, src):
    #reshapes matrcies for forward_encoder
    m = torch.reshape(m, (self.layers, 2, src.size(1), self.hidden_size // 2))
    m = torch.transpose(m, 1, 2)
    m = torch.reshape(m, (self.layers, src.size(1), self.hidden_size))
    return m 
    
  def forward_encoder(self, src, src_lengths):
    """
    Encodes source words `src`.
    Arguments:
        src: src batch of size (max_src_len, bsz)
        src_lengths: src lengths of size (bsz)
    Returns:
        memory_bank: a tensor of size (src_len, bsz, hidden_size)
        (final_state, context): `final_state` is a tuple (h, c) where h/c is of size 
                                (layers, bsz, hidden_size), and `context` is `None`. 
    """
    tokens = self.word_embeddings_src(src)
    tokens = pack(tokens, src_lengths.tolist(), batch_first = False)
    memory_bank, (h_n, c) = self.encoder_rnn(tokens)
    h_n = self.matrix_reshape(h_n, src)
    c = self.matrix_reshape(c, src)
    
    final_state = (h_n, c)
    context = None
    unpacked = unpack(memory_bank)
    return unpacked[0], (final_state, context)


  def forward_decoder(self, encoder_final_state, tgt_in, memory_bank, src_mask):
    """
    Decodes based on encoder final state, memory bank, src_mask, and ground truth 
    target words.
    Arguments:
        encoder_final_state: (final_state, None) where final_state is the encoder
                             final state used to initialize decoder. None is the
                             initial context (there's no previous context at the
                             first step).
        tgt_in: a tensor of size (tgt_len, bsz)
        memory_bank: a tensor of size (src_len, bsz, hidden_size), encoder outputs 
                     at every position
        src_mask: a tensor of size (src_len, bsz): a boolean tensor, `False` where
                  src is padding (we disallow decoder to attend to those places).
    Returns:
        Logits of size (tgt_len, bsz, V_tgt) (before the softmax operation)
    """
    max_tgt_length = tgt_in.size(0)
    
    # Initialize decoder state, note that it's a tuple (state, context) here
    decoder_states = encoder_final_state
    
    all_logits = []
    for i in range(max_tgt_length):
      logits, decoder_states, attn = \
        self.forward_decoder_incrementally(decoder_states, 
                                           tgt_in[i], 
                                           memory_bank,
                                           src_mask,
                                           normalize=False)
      all_logits.append(logits)             # list of bsz, vocab_tgt
    all_logits = torch.stack(all_logits, 0) # tgt_len, bsz, vocab_tgt
    return all_logits

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
    memory_bank, encoder_final_state = self.forward_encoder(src, src_lengths)
    # Forward decoder
    logits = self.forward_decoder(encoder_final_state, tgt_in, memory_bank, src_mask)
    return logits

  def forward_decoder_incrementally(self, prev_decoder_states, tgt_in_onestep, 
                                    memory_bank, src_mask,
                                    normalize=True):
    """
    Forward the decoder for a single step with token `tgt_in_onestep`.
    This function will be used both in `forward_decoder` and in beam search.
    Note that bsz can be greater than 1.
    Arguments:
        prev_decoder_states: a tuple (prev_decoder_state, prev_context). `prev_context`
                             is `None` for the first step
        tgt_in_onestep: a tensor of size (bsz), tokens at one step
        memory_bank: a tensor of size (src_len, bsz, hidden_size), encoder outputs 
                     at every position
        src_mask: a tensor of size (src_len, bsz): a boolean tensor, `False` where
                  src is padding (we disallow decoder to attend to those places).
        normalize: use log_softmax to normalize or not. Beam search needs to normalize,
                   while `forward_decoder` does not
    Returns:
        logits: log probabilities for `tgt_in_token` of size (bsz, V_tgt)
        decoder_states: (`decoder_state`, `context`) which will be used for the 
                        next incremental update
        attn: normalized attention scores at this step (bsz, src_len)
    """
    prev_decoder_state, prev_context = prev_decoder_states
    
    # Compute word embeddings
    tgt_embeddings = self.word_embeddings_tgt(tgt_in_onestep) # tgt_len, bsz, hidden
    
    if prev_context is not None:
        tgt_embeddings = tgt_embeddings + prev_context
    
    #resize embeddings
    bsz = tgt_in_onestep.size(0) #bsz = batchsize
    tgt_embeddings = tgt_embeddings.reshape(1, tgt_embeddings.shape[0], tgt_embeddings.shape[1])
    
    # Forward decoder RNN and return all hidden states
    decoder_outs, decoder_state = self.decoder_rnn(tgt_embeddings, prev_decoder_state)
    
    #resize src_mask
    src_mask = src_mask.transpose(0,1)
    decoder_size = decoder_outs.size(0)
    memory_size = memory_bank.size(0)
    src_mask = src_mask.reshape(bsz, decoder_size, memory_size)
    
    attn, context = attention(decoder_outs, memory_bank, memory_bank, src_mask)
    
    decoder_outs = torch.cat((decoder_outs, context), dim = -1)
    context = context.reshape(bsz, self.hidden_size)
    attn = torch.flatten(attn, start_dim=1)
    
    # Project to get logits
    logits = self.hidden2output(decoder_outs) 
    
    decoder_states = (decoder_state, context)

    if normalize:
      logits = torch.log_softmax(logits, dim=-1)
    return logits, decoder_states, attn

  def evaluate_ppl(self, iterator):
    """Returns the model's perplexity on a given dataset `iterator`."""
    # Switch to eval mode
    self.eval()
    total_loss = 0
    total_words = 0
    for batch in tqdm(iterator): # Add tqdm - MONI
      # Input and target
      src, src_lengths = batch.src
      tgt = batch.tgt # max_length_sql, bsz
      tgt_in = tgt[:-1] # remove <eos> for decode input (y_0=<bos>, y_1, y_2)
      tgt_out = tgt[1:] # remove <bos> as target        (y_1, y_2, y_3=<eos>)
      # Forward to get logits
      logits = self.forward(src, src_lengths, tgt_in)
      # Compute cross entropy loss
      loss = self.loss_function(logits.view(-1, self.V_tgt), tgt_out.view(-1))
      total_loss += loss.item()
      total_words += tgt_out.ne(self.padding_id_tgt).float().sum().item()
    return math.exp(total_loss/total_words)

  def train_all(self, train_iter, val_iter, epochs=10, learning_rate=0.001):
    """Train the model."""
    # Switch the module to training mode
    self.train()
    # Use Adam to optimize the parameters
    optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
    best_validation_ppl = float('inf')
    best_model = None
    val_ppls = [] # MONI ADDITION
    # Run the optimization for multiple epochs
    for epoch in range(epochs): 
      total_words = 0
      total_loss = 0.0
      for batch in tqdm(train_iter):
        # Zero the parameter gradients
        self.zero_grad()
        # Input and target
        src, src_lengths = batch.src # text: max_src_length, bsz
        tgt = batch.tgt # max_tgt_length, bsz
        tgt_in = tgt[:-1] # Remove <eos> for decode input (y_0=<bos>, y_1, y_2)
        tgt_out = tgt[1:] # Remove <bos> as target        (y_1, y_2, y_3=<eos>)
        bsz = tgt.size(1)
        # Run forward pass and compute loss along the way.
        logits = self.forward(src, src_lengths, tgt_in)
        loss = self.loss_function(logits.view(-1, self.V_tgt), tgt_out.view(-1))
        # Training stats
        num_tgt_words = tgt_out.ne(self.padding_id_tgt).float().sum().item()
        total_words += num_tgt_words
        total_loss += loss.item()
        # Perform backpropagation
        loss.div(bsz).backward()
        optim.step()

      # Evaluate and track improvements on the validation dataset
      validation_ppl = self.evaluate_ppl(val_iter)
      val_ppls.append(validation_ppl)
      self.train()
      if validation_ppl < best_validation_ppl:
        best_validation_ppl = validation_ppl
        self.best_model = copy.deepcopy(self.state_dict())
      epoch_loss = total_loss / total_words
      print (f'Epoch: {epoch} Training Perplexity: {math.exp(epoch_loss):.4f} '
             f'Validation Perplexity: {validation_ppl:.4f}')

    return val_ppls

def main():
    EPOCHS = 2 # epochs, we highly recommend starting with a smaller number like 1
    LEARNING_RATE = 2e-3 # learning rate

    # Instantiate and train classifier
    model = AttnEncoderDecoder(SRC, TGT,
                               hidden_size    = 64,
                               layers         = 3,
                           ).to(device)
    
    # train_iter, val_iter = train_iter.to(device), val_iter.to(device)
    
    start_time = time()
    model.train_all(train_iter, val_iter, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    model.load_state_dict(model.best_model)
    
    print("AttnEncoderDecoder took {} seconds to train.".format(time() - start_time))

if __name__ == '__main__':
    main()




