import torch
import torch.nn as nn


class LSTMLayer(nn.Module):

  def __init__(self, input_size, hidden_size, dtype=torch.float, device='cpu'):
    super(LSTMLayer, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.factory_kwargs = {'dtype': dtype, 'device': device}

    # cell_size == hidden_size
    self.linear_write = nn.Linear(input_size + 2 * hidden_size, hidden_size,
                                  **self.factory_kwargs)
    self.linear_forget = nn.Linear(input_size + 2 * hidden_size, hidden_size,
                                  **self.factory_kwargs)
    self.linear_cell = nn.Linear(input_size + hidden_size, hidden_size,
                                  **self.factory_kwargs)
    self.linear_output = nn.Linear(input_size + 2 * hidden_size, hidden_size,
                                  **self.factory_kwargs)

  def forward(self, input, states=None):
    """Args:
        input: torch.Tensor, [seq_len, input_size] or
          [seq_len, batch_size, input_size]
        states (optional): a tuple of two torch.Tensor
            hidden: torch.Tensor, [hidden_size]  or [batch_size, hidden_size]
            cell: torch.Tensor, [hidden_size] or [batch_size, hidden_size]

    Return:
        output: torch.Tensor, [seq_len, hidden_size] or
            [seq_len, batch_size, hidden_size]
        states: a tuple of two torch.Tensor
            hidden: torch.Tensor, [hidden_size] or [batch_size, hidden_size]
            cell: torch.Tensor, [hidden_size] or [batch_size, hidden_size]
    """
    assert (2 <= len(input.shape) <= 3) and input.size(-1) == self.input_size, \
      "The shape of the `input` should be [seq_len, input_size] or " \
      "[seq_len, batch_size, input_size]"

    is_batched = len(input.shape) == 3
    if is_batched:
      seq_len, batch_size, _ = input.shape
      outputs = torch.zeros(seq_len, batch_size, self.hidden_size,
                            **self.factory_kwargs)
      if states is None:
        hidden = torch.zeros(batch_size, self.hidden_size,
                             **self.factory_kwargs)
        cell = torch.zeros(batch_size, self.hidden_size, **self.factory_kwargs)
      else:
        hidden, cell = states
    else:
      seq_len, _ = input.shape
      outputs = torch.zeros(seq_len, self.hidden_size, **self.factory_kwargs)
      if states is None:
        hidden = torch.zeros(self.hidden_size, **self.factory_kwargs)
        cell = torch.zeros(self.hidden_size, **self.factory_kwargs)
      else:
        hidden, cell = states

    assert (1 <= len(hidden.shape) <= 2) and \
      hidden.size(-1) == self.hidden_size, \
      "The shape of the `hidden` should be [hidden_size] or " \
      "[batch_size, hidden_size]"
    assert (1 <= len(cell.shape) <= 2) and \
      cell.size(-1) == self.hidden_size, \
      "The shape of the `cell` should be [hidden_size] or " \
      "[batch_size, hidden_size]"
    
    seq_len = input.size(0)
    for i in range(seq_len):
      # input becomes [input_size] or [batch_size, input_size]
      combined = torch.cat((input[i], hidden, cell), dim=len(input[i].shape)-1)
      write = torch.sigmoid(self.linear_write(combined))
      forget = torch.sigmoid(self.linear_forget(combined))

      combined = torch.cat((input[i], hidden), dim=len(input[i].shape)-1)
      cell = forget * cell + write * torch.tanh(self.linear_cell(combined))
      
      combined = torch.cat((input[i], hidden, cell),
                           dim=len(input[i].shape)-1)
      output = torch.sigmoid(self.linear_output(combined))
      hidden = output * torch.tanh(cell)
      outputs[i] = hidden

    return outputs, (hidden, cell)


class LSTM(nn.Module):

  def __init__(self, input_size, hidden_size, num_layers, dtype=torch.float, device='cpu'):
    super(LSTM, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.factory_kwargs = {'dtype': dtype, 'device': device}

    layers = [LSTMLayer(input_size, hidden_size, **self.factory_kwargs)] + \
      [LSTMLayer(hidden_size, hidden_size, **self.factory_kwargs)
      for _ in range(num_layers - 1)]
    self.layers = nn.ModuleList(layers)

  def forward(self, input, states=None):
    """Args:
        input: torch.Tensor, [seq_len, input_size] or
          [seq_len, batch_size, input_size]
        states (optional): a tuple of two torch.Tensor
            hidden: torch.Tensor, [num_layers, hidden_size] or
                [num_layers, batch_size, hidden_size]
            cell: torch.Tensor, [num_layers, hidden_size] or
                [num_layers, batch_size, hidden_size]

    Return:
        output: torch.Tensor, [seq_len, hidden_size] or
            [seq_len, batch_size, hidden_size]
        states: a tuple of two torch.Tensor
            hidden: torch.Tensor, [num_layers, hidden_size] or
                [num_layers, batch_size, hidden_size]
            cell: torch.Tensor, [num_layers, hidden_size] or
                [num_layers, batch_size, hidden_size]
    """
    assert (2 <= len(input.shape) <= 3) and input.size(-1) == self.input_size, \
      "The shape of the `input` should be [seq_len, input_size] or " \
      "[seq_len, batch_size, input_size]"

    is_batched = len(input.shape) == 3
    if is_batched:
      seq_len, batch_size, _ = input.shape
      if states is None:
        hiddens = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                             **self.factory_kwargs)
        cells = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                           **self.factory_kwargs)
      else:
        hiddens, cells = states
    else:
      seq_len, _ = input.shape
      if states is None:
        hiddens = torch.zeros(self.num_layers, self.hidden_size,
                             **self.factory_kwargs)
        cells = torch.zeros(self.num_layers, self.hidden_size,
                           **self.factory_kwargs)
      else:
        hiddens, cells = states

    assert (2 <= len(hiddens.shape) <= 3) and \
      hiddens.size(0) == self.num_layers and \
      hiddens.size(-1) == self.hidden_size, \
      "The shape of the `hidden` should be [num_layers, hidden_size] or " \
      "[num_layers, batch_size, hidden_size]"
    assert (2 <= len(cells.shape) <= 3) and \
      cells.size(0) == self.num_layers and \
      cells.size(-1) == self.hidden_size, \
      "The shape of the `cell` should be [num_layers, hidden_size] or " \
      "[num_layers, batch_size, hidden_size]"
    
    next_hiddens = torch.zeros_like(hiddens)
    next_cells = torch.zeros_like(cells)
    
    output = input
    for i in range(self.num_layers):
      # hidden and cell are [hidden_size] or [batch_size, hidden_size]
      output, (hidden, cell) = self.layers[i](output, (hiddens[i], cells[i]))
      next_hiddens[i], next_cells[i] = hidden, cell

    return output, (next_hiddens, next_cells)


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
    self.rnn = LSTM(embed_size, hidden_size, num_rnn_layers,
                    **self.factory_kwargs)

  def forward(self, input, states=None):
    """Args:
        input: torch.Tensor, [seq_len] or [seq_len, batch_size]
        states (optional): a tuple of two torch.Tensor
            hidden: torch.Tensor, [num_rnn_layers, hidden_size] or
                [num_rnn_layers, batch_size, hidden_size]
            cell: torch.Tensor, [num_rnn_layers, hidden_size] or
                [num_rnn_layers, batch_size, hidden_size]

    Return:
        output: torch.Tensor, [seq_len, hidden_size] or
            [seq_len, batch_size, hidden_size]
        states: a tuple of two torch.Tensor
            hidden: torch.Tensor, [num_rnn_layers, hidden_size] or
                [num_rnn_layers, batch_size, hidden_size]
            cell: torch.Tensor, [num_rnn_layers, hidden_size] or
                [num_rnn_layers, batch_size, hidden_size]
    """
    embedded = self.embedding(input)
    output, (hidden, cell) = \
      self.rnn(embedded) if states is None else self.rnn(embedded, states)
    return output, (hidden, cell)


class Decoder(nn.Module):

  def __init__(self, embed_size, hidden_size, output_size, num_rnn_layers,
               padding_index, dtype=torch.float, device='cpu'):
    super(Decoder, self).__init__()
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.num_rnn_layers = num_rnn_layers
    self.factory_kwargs = {'dtype': dtype, 'device': device}

    input_size = output_size
    self.embedding = nn.Embedding(input_size, embed_size, padding_index,
                                  **self.factory_kwargs)
    self.rnn = LSTM(embed_size, hidden_size, num_rnn_layers,
                    **self.factory_kwargs)
    self.linear = nn.Linear(hidden_size, output_size, **self.factory_kwargs)
    # linear layer가 들어가는게 맞나?

  def forward(self, input, states=None, beam_size=1, max_len=50,
              teacher_forcing_ratio=0.):
    """Args:
        input: torch.Tensor, [seq_len] or [seq_len, batch_size]
        states (optional): a tuple of two torch.Tensor
            hidden: torch.Tensor, [num_layers, hidden_size] or
                [num_layers, batch_size, hidden_size]
            cell: torch.Tensor, [num_layers, hidden_size] or
                [num_layers, batch_size, hidden_size]
        beam_size (optional): a non-negative integer
        max_len (optional): a non-negative integer
        teacher_forcing_ratio (optional): a float number between 0 and 1

    Return:
        output: torch.Tensor, [max_len, output_size] or
            [max_len, batch_size, output_size]
        states: a tuple of two torch.Tensor
            hidden: torch.Tensor, [num_layers, hidden_size] or
                [num_layers, batch_size, hidden_size]
            cell: torch.Tensor, [num_layers, hidden_size] or
                [num_layers, batch_size, hidden_size]
    """
    #TODO: sample until all rows have more than one EOS
    #TODO: forward with beam search 구현
    use_beam_search = beam_size != 1
    if use_beam_search:
      raise NotImplementedError()
    else:
      # input.size(0) == target length
      if self.training: max_len = input.size(0)
      
      is_batched = len(input.shape) == 2
      if is_batched:
        outputs = torch.zeros(max_len, input.size(1), self.output_size,
                             **self.factory_kwargs)
      else:
        outputs = torch.zeros(max_len, self.output_size, **self.factory_kwargs)

      assert states is not None, "You should give hidden states and cell " \
        "states into the decoder"
      hidden, cell = states

      inputs = input
      input_shape = (1, input.size(1)) if is_batched else (1,)
      input = inputs[0].view(input_shape) # [1] or [1, batch_size]
      for i in range(1, max_len):
        embedded = self.embedding(input)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        output = self.linear(output)
        outputs[i] = output.view(outputs.shape[1:])
        if self.training and torch.randn(1) < teacher_forcing_ratio:
          # use teacher forcing
          input = inputs[i].view(input_shape)
        else:
          # do not use teacher forcing
          input = output.argmax(len(inputs.shape)).view(input_shape)
          
      return outputs, (hidden, cell)


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

  def forward(self, src, trg, beam_size=1, max_len=50,
              teacher_forcing_ratio=0.):
    """Args:
        src: torch.Tensor, [src_len] or [src_len, batch_size]
        trg: torch.Tensor, [trg_len] or [trg_len, batch_size]
        beam_size (optional): a non-negative integer
        max_len (optional): a non-negative integer
        teacher_forcing_ratio (optional): a float number between 0 and 1

    Return:
        output: torch.Tensor, [trg_len, output_size] or
            [trg_len, batch_size, output_size]
    """
    _, (hidden, cell) = self.encoder(src)
    output, _ = self.decoder(trg, (hidden, cell), beam_size, max_len,
                             teacher_forcing_ratio=teacher_forcing_ratio)
    return output

  def encode(self, input, states=None):
    """Args:
        input: torch.Tensor, [seq_len] or [seq_len, batch_size]
        states (optional): a tuple of two torch.Tensor
            hidden: torch.Tensor, [num_layers, hidden_size] or
                [num_layers, batch_size, hidden_size]
            cell: torch.Tensor, [num_layers, hidden_size] or
                [num_layers, batch_size, hidden_size]

    Return:
        output: torch.Tensor, [seq_len, hidden_size] or
            [trg_len, batch_size, hidden_size]
        states (optional): a tuple of two torch.Tensor
            hidden: torch.Tensor, [num_layers, hidden_size] or
                [num_layers, batch_size, hidden_size]
            cell: torch.Tensor, [num_layers, hidden_size] or
                [num_layers, batch_size, hidden_size]
    """
    return self.encoder(input, states)

  def decode(self, input, states=None, beam_size=1, max_len=50,
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
    output, _ = self.decoder(input, states, beam_size, max_len,
                             teacher_forcing_ratio)
    return output
