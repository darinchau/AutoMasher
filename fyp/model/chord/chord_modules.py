import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)
    return torch_mask.unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


class LayerNorm(nn.Module):
    # Borrowed from jekbradbury
    # https://github.com/pytorch/pytorch/issues/1959
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class OutputLayer(nn.Module):
    """
    Abstract base class for output layer.
    Handles projection to output labels
    """

    def __init__(self, hidden_size, output_size, probs_out=False):
        super(OutputLayer, self).__init__()
        self.output_size = output_size
        self.output_projection = nn.Linear(hidden_size, output_size)
        self.probs_out = probs_out
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=int(hidden_size/2), batch_first=True, bidirectional=True)
        self.hidden_size = hidden_size

    def loss(self, hidden, labels):
        raise NotImplementedError('Must implement {}.loss'.format(self.__class__.__name__))


class SoftmaxOutputLayer(OutputLayer):
    """
    Implements a softmax based output layer
    """

    def forward(self, hidden):
        logits = self.output_projection(hidden)
        probs = F.softmax(logits, -1)
        _, predictions = torch.max(probs, dim=-1)
        return predictions, logits

    def loss(self, hidden, labels):
        logits = self.output_projection(hidden)
        log_probs = F.log_softmax(logits, -1)
        return F.nll_loss(log_probs.view(-1, self.output_size), labels.view(-1))


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """

    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth,
                 num_heads, bias_mask=None, dropout=0.0, attention_map=False):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
        if total_key_depth % num_heads != 0:
            raise ValueError("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_depth, num_heads))
        if total_value_depth % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_depth, num_heads))

        self.attention_map = attention_map

        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5
        self.bias_mask = bias_mask

        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3] * self.num_heads)

    def forward(self, queries, keys, values):

        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        # Add bias to mask future values
        if self.bias_mask is not None:
            logits += self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits.data)

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)

        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)

        # Merge heads
        contexts = self._merge_heads(contexts)
        # contexts = torch.tanh(contexts)

        # Linear to get output
        outputs = self.output_linear(contexts)

        if self.attention_map is True:
            return outputs, weights

        return outputs


class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """

    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data_loader),
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = (kernel_size - 1, 0) if pad_type == 'left' else (kernel_size // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)

        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """

    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data_loader),
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()

        layers = []
        sizes = ([(input_depth, filter_size)] +
                 [(filter_size, filter_size)] * (len(layer_config) - 2) +
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0, attention_map=False):
        super(SelfAttentionBlock, self).__init__()

        self.attention_map = attention_map
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, hidden_size, num_heads, bias_mask, attention_dropout, attention_map)
        self.positionwise_convolution = PositionwiseFeedForward(hidden_size, filter_size, hidden_size, layer_config='cc', padding='both', dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        x = inputs

        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        if self.attention_map is True:
            y, weights = self.multi_head_attention(x_norm, x_norm, x_norm)
        else:
            y = self.multi_head_attention(x_norm, x_norm, x_norm)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_convolution(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        if self.attention_map is True:
            return y, weights
        return y


class BidirectionalSelfAttention(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, max_length,
                 layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):

        super(BidirectionalSelfAttention, self).__init__()

        self.weights_list = list()

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),
                  layer_dropout,
                  attention_dropout,
                  relu_dropout,
                  True)

        self.attn_block = SelfAttentionBlock(*params)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  torch.transpose(_gen_bias_mask(max_length), dim0=2, dim1=3),
                  layer_dropout,
                  attention_dropout,
                  relu_dropout,
                  True)

        self.backward_attn_block = SelfAttentionBlock(*params)

        self.linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, inputs):
        x, list = inputs

        # Forward Self-attention Block
        encoder_outputs, weights = self.attn_block(x)
        # Backward Self-attention Block
        reverse_outputs, reverse_weights = self.backward_attn_block(x)
        # Concatenation and Fully-connected Layer
        outputs = torch.cat((encoder_outputs, reverse_outputs), dim=2)
        y = self.linear(outputs)

        # Attention weights for Visualization
        self.weights_list = list
        self.weights_list.append(weights)
        self.weights_list.append(reverse_weights)
        return y, self.weights_list


class BidirectionalSelfAttentionLayers(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0):
        super(BidirectionalSelfAttentionLayers, self).__init__()

        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  max_length,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.self_attn_layers = nn.Sequential(*[BidirectionalSelfAttention(*params) for l in range(num_layers)])
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        # A Stack of Bi-directional Self-attention Layers
        y, weights_list = self.self_attn_layers((x, []))

        # Layer Normalization
        y = self.layer_norm(y)
        return y, weights_list


class BTCModel(nn.Module):
    def __init__(self, config):
        super(BTCModel, self).__init__()

        self.timestep = config['timestep']
        self.probs_out = config['probs_out']

        params = (config['feature_size'],
                  config['hidden_size'],
                  config['num_layers'],
                  config['num_heads'],
                  config['total_key_depth'],
                  config['total_value_depth'],
                  config['filter_size'],
                  config['timestep'],
                  config['input_dropout'],
                  config['layer_dropout'],
                  config['attention_dropout'],
                  config['relu_dropout'])

        self.self_attn_layers = BidirectionalSelfAttentionLayers(*params)
        self.output_layer = SoftmaxOutputLayer(hidden_size=config['hidden_size'], output_size=config['num_chords'], probs_out=config['probs_out'])

    def forward(self, x, labels):
        labels = labels.view(-1, self.timestep)
        # Output of Bi-directional Self-attention Layers
        self_attn_output, weights_list = self.self_attn_layers(x)

        # return logit values for CRF
        if self.probs_out is True:
            logits = self.output_layer(self_attn_output)
            return logits

        # Output layer and Soft-max
        prediction, second = self.output_layer(self_attn_output)
        prediction = prediction.view(-1)
        second = second.view(-1)

        # Loss Calculation
        loss = self.output_layer.loss(self_attn_output, labels)
        return prediction, loss, weights_list, second
