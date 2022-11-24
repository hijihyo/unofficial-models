import torch
import torch.nn as nn

class GRULayer(nn.Module):

  def __init__(self, input_size, hidden_size, is_decoder, dtype=torch.float, device='cpu'):
    super(GRULayer, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.is_decoder = is_decoder
    self.factory_kwargs = {'dtype': dtype, 'device': device}

    # summary_size == hidden_size
    combined_size = input_size + 2 * hidden_size if is_decoder \
      else input_size + hidden_size
    self.linear_reset = nn.Linear(combined_size, hidden_size,
                                  **self.factory_kwargs)
    self.linear_update = nn.Linear(combined_size, hidden_size,
                                   **self.factory_kwargs)
    self.linear_new = nn.Linear(combined_size, hidden_size,
                                **self.factory_kwargs)

  def forward(self, input, hidden=None, summary=None):
    """Args:
        input: torch.Tensor, [seq_len, input_size] or
          [seq_len, batch_size, input_size]
        hidden: torch.Tensor, [hidden_size] or [batch_size, hidden_size]
        summary: torch.Tensor, [hidden_size] or [batch_size, hidden_size]

    Return:
        output: torch.Tensor, [seq_len, hidden_size] or
            [seq_len, batch_size, hidden_size]
        hidden: torch.Tensor, [hidden_size] or [batch_size, hidden_size]
    """
    assert (2 <= len(input.shape) <= 3) and input.size(-1) == self.input_size, \
      "The shape of the `input` should be [seq_len, input_size] or " \
      "[seq_len, batch_size, input_size]"
    assert (not self.is_decoder and summary is None) or \
      (self.is_decoder and hidden is not None and summary is not None), \
      "The GRU for an encoder should not receive a summary vector and for " \
      "a decoder should receive a hidden state and a summary vector."
    assert (hidden is None) or \
      (len(hidden.shape) == len(input.shape) - 1 and \
       hidden.size(-1) == self.hidden_size), \
      "The shape of the `hidden` should be [hidden_size] or " \
      "[batch_size, hidden_size]"
    assert (summary is None) or \
      (len(summary.shape) == len(input.shape) - 1 and \
       summary.size(-1) == self.hidden_size), \
      "The shape of the `summary` should be [hidden_size] or " \
      "[batch_size, hidden_size]"
    
    is_batched = len(input.shape) == 3
    if is_batched:
      seq_len, batch_size, _ = input.shape
      outputs = torch.zeros(seq_len, batch_size, self.hidden_size,
                            **self.factory_kwargs)
      if hidden is None:
        hidden = torch.zeros(batch_size, self.hidden_size,
                             **self.factory_kwargs)
    else:
      seq_len, _ = input.shape
      outputs = torch.zeros(seq_len, self.hidden_size,
                            **self.factory_kwargs)
      if hidden is None:
        hidden = torch.zeros(self.hidden_size,
                             **self.factory_kwargs)
    
    for i in range(seq_len):
      if self.is_decoder:
        combined = torch.cat((input[i], hidden, summary),
                             dim=len(input[i].shape)-1)
      else:
        combined = torch.cat((input[i], hidden), dim=len(input[i].shape)-1)
      reset = torch.sigmoid(self.linear_reset(combined))
      update = torch.sigmoid(self.linear_update(combined))

      if self.is_decoder:
        combined = torch.cat((input[i], reset * hidden, reset * summary),
                             dim=len(input[i].shape)-1)
      else:
        combined = torch.cat((input[i], reset * hidden),
                             dim=len(input[i].shape)-1)
      new = torch.tanh(self.linear_new(combined))
      hidden = update * hidden + (1 - update) * new

      outputs[i] = hidden
    
    return outputs, hidden


class GRU(nn.Module):

  def __init__(self, input_size, hidden_size, num_layers, is_decoder, dtype=torch.float, device='cpu'):
    super(GRU, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.is_decoder = is_decoder
    self.factory_kwargs = {'dtype': dtype, 'device': device}

    layers = \
      [GRULayer(input_size, hidden_size, is_decoder, **self.factory_kwargs)] + \
      [GRULayer(hidden_size, hidden_size, is_decoder, **self.factory_kwargs)
      for _ in range(num_layers - 1)]
    self.layers = nn.ModuleList(layers)

  def forward(self, input, hiddens=None, summarys=None):
    """Args:
        input: torch.Tensor, [seq_len, input_size] or
          [seq_len, batch_size, input_size]
        hiddens: torch.Tensor, [num_layers, hidden_size] or
          [num_layers, batch_size, hidden_size]
        summarys: torch.Tensor, [num_layers, hidden_size] or
          [num_layers, batch_size, hidden_size]

    Return:
        output: torch.Tensor, [seq_len, hidden_size] or
            [seq_len, batch_size, hidden_size]
        hidden: torch.Tensor, [num_layers, hidden_size] or
          [num_layers, batch_size, hidden_size]
    """
    assert (2 <= len(input.shape) <= 3) and input.size(-1) == self.input_size, \
      "The shape of the `input` should be [seq_len, input_size] or " \
      "[seq_len, batch_size, input_size]"
    assert (not self.is_decoder and summarys is None) or \
      (self.is_decoder and hiddens is not None and summarys is not None), \
      "The GRU for an encoder should not receive a summary vector and for " \
      "a decoder should receive a hidden state and a summary vector."
    assert (hiddens is None) or \
      (len(hiddens.shape) == len(input.shape) and \
       hiddens.size(0) == self.num_layers and \
       hiddens.size(-1) == self.hidden_size), \
      "The shape of the `hidden` should be [num_layers, hidden_size] or " \
      "[num_layers, batch_size, hidden_size]"
    assert (summarys is None) or \
      (len(summarys.shape) == len(input.shape) and \
       summarys.size(0) == self.num_layers and \
       summarys.size(-1) == self.hidden_size), \
      "The shape of the `summary` should be [num_layers, hidden_size] or " \
      "[num_layers, batch_size, hidden_size]"

    is_batched = len(input.shape) == 3
    if is_batched:
      seq_len, batch_size, _ = input.shape
      if hiddens is None:
        hiddens = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                              **self.factory_kwargs)
    else:
      seq_len, _ = input.shape
      if hiddens is None:
        hiddens = torch.zeros(self.num_layers, self.hidden_size,
                              **self.factory_kwargs)

    output = input
    next_hiddens = torch.zeros_like(hiddens)
    for i in range(self.num_layers):
      if self.is_decoder:
        output, hidden = self.layers[i](output, hiddens[i], summarys[i])
      else:
        output, hidden = self.layers[i](output, hiddens[i])
      next_hiddens[i] = hidden

    return output, next_hiddens


class Encoder(nn.Module):

  def __init__(self, input_size, embed_size, hidden_size, num_rnn_layers,
               padding_index, dtype=torch.float, device='cpu'):
    super(Encoder, self).__init__()
    self.input_size = input_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.num_rnn_layers = num_rnn_layers
    self.factory_kwargs = {'dtype': dtype, 'device': device}

    self.embedding = nn.Embedding(input_size, embed_size, padding_index,
                                  **self.factory_kwargs)
    self.rnn = GRU(embed_size, hidden_size, num_rnn_layers, is_decoder=False,
                    **self.factory_kwargs)
    self.linear_summary = nn.Linear(hidden_size, hidden_size,
                                    **self.factory_kwargs)

  def forward(self, input, hidden=None):
    """Args:
        input: torch.Tensor, [seq_len] or [seq_len, batch_size]
        hidden (optional): torch.Tensor, [num_rnn_layers, hidden_size] or
          [num_rnn_layers, batch_size, hidden_size]

    Return:
        output: torch.Tensor, [seq_len, hidden_size] or
            [seq_len, batch_size, hidden_size]
        hidden: torch.Tensor, [num_rnn_layers, hidden_size] or
          [num_rnn_layers, batch_size, hidden_size]
        summary: torch.Tensor, [num_rnn_layers, hidden_size] or
          [num_rnn_layers, batch_size, hidden_size]
    """
    embedded = self.embedding(input)
    output, hidden = self.rnn(embedded, hidden)
    summary = torch.tanh(self.linear_summary(hidden))
    return output, hidden, summary


class Decoder(nn.Module):

  def __init__(self, embed_size, hidden_size, output_size, num_rnn_layers,
               padding_index, dtype=torch.float, device='cpu'):
    super(Decoder, self).__init__()
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.num_rnn_layers = num_rnn_layers
    self.num_maxouts = 500
    self.pool_size = 2
    self.stride = 2
    self.factory_kwargs = {'dtype': dtype, 'device': device}

    input_size = output_size
    self.embedding = nn.Embedding(input_size, embed_size, padding_index,
                                  **self.factory_kwargs)
    self.linear_hidden = nn.Linear(hidden_size, hidden_size,
                                   **self.factory_kwargs)
    self.rnn = GRU(embed_size, hidden_size, num_rnn_layers, is_decoder=True,
                    **self.factory_kwargs)
    # 아래에서 input_size 대신 embed_size를 했는데 괜찮을까?
    self.linear_maxout = nn.Linear(embed_size + 2 * hidden_size,
                                    self.num_maxouts * self.pool_size,
                                    **self.factory_kwargs)
    self.linear_output = nn.Linear(self.num_maxouts, output_size,
                                   **self.factory_kwargs)

  def forward(self, input, hidden=None, summary=None, max_len=50,
              teacher_forcing_ratio=0.):
    """Args:
        input: torch.Tensor, [seq_len] or [seq_len, batch_size]
        hidden: torch.Tensor, [num_layers, hidden_size] or
          [num_layers, batch_size, hidden_size]
        summary: torch.Tensor, [num_layers, hidden_size] or
          [num_layers, batch_size, hidden_size]
        max_len (optional): a non-negative integer
        teacher_forcing_ratio (optional): a float number between 0 and 1

    Return:
        output: torch.Tensor, [max_len, output_size] or
            [max_len, batch_size, output_size]
        hidden: torch.Tensor, [num_layers, hidden_size] or
          [num_layers, batch_size, hidden_size]
        summary: torch.Tensor, [num_layers, hidden_size] or
          [num_layers, batch_size, hidden_size]
    """
    #TODO: sample until all rows have more than one EOS
    # input.size(0) == target length
    if self.training: max_len = input.size(0)
      
    is_batched = len(input.shape) == 2
    if is_batched:
      _, batch_size = input.shape
      outputs = torch.zeros(max_len, batch_size, self.output_size,
                            **self.factory_kwargs)
    else:
      outputs = torch.zeros(max_len, self.output_size, **self.factory_kwargs)

    assert summary is not None, "You should give summary vector into the " \
      "decoder"
    if hidden is None:
      hidden = torch.tanh(self.linear_hidden(summary))

    inputs = input
    input_shape = (1, batch_size) if is_batched else (1,)
    input = inputs[0].view(input_shape) # [1] or [1, batch_size]
    for i in range(1, max_len):
      embedded = self.embedding(input)
      output, hidden = self.rnn(embedded, hidden, summary)
      combined = torch.cat((hidden[-1], embedded[0], summary[-1]),
                            dim=len(hidden.shape)-2)
      # [batch_size, embed_size + 2 * hidden_size]
      # -> [batch_size, self.num_maxouts]
      maxout = nn.functional.max_pool1d(self.linear_maxout(combined),
                                        kernel_size=self.pool_size,
                                        stride=self.stride)
      output = self.linear_output(maxout) # [batch_size, output_size]
      outputs[i] = output.view(outputs.shape[1:])
      if self.training and torch.randn(1) < teacher_forcing_ratio:
        # use teacher forcing
        input = inputs[i].view(input_shape)
      else:
        # do not use teacher forcing
        input = output.argmax(len(input.shape)-1).view(input_shape)
          
    return outputs, hidden, summary


class Seq2SeqNetwork(nn.Module):

  def __init__(self, input_size, embed_size, hidden_size, output_size,
               num_rnn_layers, padding_index, dtype=torch.float, device='cpu'):
    super(Seq2SeqNetwork, self).__init__()
    self.input_size = input_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.num_rnn_layers = num_rnn_layers
    self.factory_kwargs = {'dtype': dtype, 'device': device}

    self.encoder = Encoder(input_size, embed_size, hidden_size, num_rnn_layers,
                           padding_index, **self.factory_kwargs)
    self.decoder = Decoder(embed_size, hidden_size, output_size, num_rnn_layers,
                           padding_index, **self.factory_kwargs)

  def forward(self, src, trg, max_len=50, teacher_forcing_ratio=0.):
    """Args:
        src: torch.Tensor, [src_len] or [src_len, batch_size]
        trg: torch.Tensor, [trg_len] or [trg_len, batch_size]
        max_len (optional): a non-negative integer
        teacher_forcing_ratio (optional): a float number between 0 and 1

    Return:
        output: torch.Tensor, [trg_len, output_size] or
            [trg_len, batch_size, output_size]
    """
    _, _, summary = self.encoder(src)
    output, _, _ = self.decoder(trg, summary=summary, max_len=max_len,
                             teacher_forcing_ratio=teacher_forcing_ratio)
    return output

  def encode(self, input, hidden=None):
    """Args:
        input: torch.Tensor, [seq_len] or [seq_len, batch_size]
        hidden: torch.Tensor, [num_layers, hidden_size] or
          [num_layers, batch_size, hidden_size]

    Return:
        output: torch.Tensor, [seq_len, hidden_size] or
            [trg_len, batch_size, hidden_size]
        hidden: torch.Tensor, [num_layers, hidden_size] or
          [num_layers, batch_size, hidden_size]
        summary: torch.Tensor, [num_layers, hidden_size] or
          [num_layers, batch_size, hidden_size]
    """
    return self.encoder(input, hidden)

  def decode(self, input, hidden=None, summary=None, beam_size=1, max_len=50,
            teacher_forcing_ratio=0.):
    """Args:
        input: torch.Tensor, [seq_len] or [seq_len, batch_size]
        beam_size (optional): a non-negative integer
        max_len (optional): a non-negative integer
        teacher_forcing_ratio (optional): a float number between 0 and 1

    Return:
        output: torch.Tensor, [max_len, output_size] or
            [max_len, batch_size, output_size]
    """
    output, _ = self.decoder(input, hidden, summary, max_len,
                             teacher_forcing_ratio)
    return output
