from typing import List

from overrides import overrides
import torch
from torch.nn import Dropout
from torch.nn import LayerNorm
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.activations import Activation
from allennlp.nn.util import add_positional_features, weighted_sum
from allennlp.common.checks import check_dimensions_match
from reading_comprehension.modules.layer_dropout import ResidualWithLayerDropout
from reading_comprehension.modules.depthwise_separable_conv import DepthwiseSeparableConv
from reading_comprehension.utils import memory_effient_masked_softmax as masked_softmax


@Seq2SeqEncoder.register("qanet_encoder_block")
class QaNetEncoderBlock(Seq2SeqEncoder):
    """
    Implements the encoder block described in `QANet: Combining Local
    Convolution with Global Self-attention for Reading Comprehension
    <https://openreview.net/forum?id=B14TlG-RW>`_ .

    One encoder block mainly contains 4 parts:

    1. Add position embedding
    2. Several depthwise seperable convolutions.
    3. Multi-headed self attention, which uses 2 learnt linear projections
       to perform a dot-product similarity between every pair of elements
       scaled by the square root of the sequence length.
    4. A two-layer FeedForward network.

    Parameters
    ----------
    input_dim : ``int``, required.
        The input dimension of the encoder.
    hidden_dim : ``int``, required.
        The hidden dimension used for convolution output channels, multi-head attention output
        and the final output of feedforward layer.
    attention_projection_dim : ``int``, required.
        The dimension of the linear projections for the self-attention layers.
    feedforward_hidden_dim : ``int``, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_convs: ``int``, required.
        The number of convolutions in each block.
    conv_kernel_size: ``int``, required.
        The kernel size for convolution.
    num_attention_heads : ``int``, required.
        The number of attention heads to use per layer.
    use_positional_encoding: ``bool``, optional, (default = True)
        Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
        as without this feature, the self attention layers have no idea of absolute or relative
        position (as they are just computing pairwise similarity between vectors of elements),
        which can be important features for many tasks.
    dropout_prob : ``float``, optional, (default = 0.1)
        The dropout probability for the feedforward network.
    layer_dropout_undecayed_prob : ``float``, optional, (default = 0.1)
        The initial dropout probability for layer dropout, and this might decay w.r.t the depth
        of the layer. For each mini-batch, the convolution/attention/ffn sublayer is
        stochastically dropped according to its layer dropout probability.
    attention_dropout_prob : ``float``, optional, (default = 0)
        The dropout probability for the attention distributions in the attention layer.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 attention_projection_dim: int,
                 feedforward_hidden_dim: int,
                 num_convs: int,
                 conv_kernel_size: int,
                 num_attention_heads: int,
                 use_positional_encoding: bool = True,
                 dropout_prob: float = 0.1,
                 layer_dropout_undecayed_prob: float = 0.1,
                 attention_dropout_prob: float = 0) -> None:
        super().__init__()

        check_dimensions_match(input_dim, hidden_dim, 'input_dim', 'hidden_dim')

        self._use_positional_encoding = use_positional_encoding

        self._conv_norm_layers = torch.nn.ModuleList([LayerNorm(hidden_dim) for _ in range(num_convs)])
        self._conv_layers = torch.nn.ModuleList([
                DepthwiseSeparableConv(hidden_dim, hidden_dim, conv_kernel_size, activation="relu", dim=1)
                for _ in range(num_convs)])

        self.attention_norm_layer = LayerNorm(hidden_dim)
        self.attention_layer = MemoryEfficientMultiHeadSelfAttention(num_heads=num_attention_heads,
                                                                     input_dim=hidden_dim,
                                                                     attention_dim=attention_projection_dim,
                                                                     values_dim=attention_projection_dim,
                                                                     attention_dropout_prob=attention_dropout_prob)
        self.feedforward_norm_layer = LayerNorm(hidden_dim)
        self.feedforward = FeedForward(hidden_dim,
                                       activations=[Activation.by_name('relu')(),
                                                    Activation.by_name('linear')()],
                                       hidden_dims=[feedforward_hidden_dim, hidden_dim],
                                       num_layers=2,
                                       dropout=dropout_prob)

        self.dropout = Dropout(dropout_prob)
        self.residual_with_layer_dropout = ResidualWithLayerDropout(layer_dropout_undecayed_prob)
        self._input_dim = input_dim
        self._output_dim = hidden_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ

        if self._use_positional_encoding:
            output = add_positional_features(inputs)
        else:
            output = inputs

        total_sublayers = len(self._conv_layers) + 2
        sublayer_count = 0

        for conv_norm_layer, conv_layer in zip(self._conv_norm_layers, self._conv_layers):
            conv_norm_out = self.dropout(conv_norm_layer(output))
            conv_out = self.dropout(conv_layer(conv_norm_out.transpose_(1, 2)).transpose_(1, 2))
            sublayer_count += 1
            output = self.residual_with_layer_dropout(output, conv_out,
                                                      sublayer_count, total_sublayers)

        attention_norm_out = self.dropout(self.attention_norm_layer(output))
        attention_out = self.dropout(self.attention_layer(attention_norm_out, mask))
        sublayer_count += 1
        output = self.residual_with_layer_dropout(output, attention_out,
                                                  sublayer_count, total_sublayers)

        feedforward_norm_out = self.dropout(self.feedforward_norm_layer(output))
        feedforward_out = self.dropout(self.feedforward(feedforward_norm_out))
        sublayer_count += 1
        output = self.residual_with_layer_dropout(output, feedforward_out,
                                                  sublayer_count, total_sublayers)

        return output


@Seq2SeqEncoder.register("qanet_encoder")
class QaNetEncoder(Seq2SeqEncoder):
    """
    Stack multiple QANetEncoderBlock into one sequence encoder.

    Parameters
    ----------
    input_dim : ``int``, required.
        The input dimension of the encoder.
    hidden_dim : ``int``, required.
        The hidden dimension used for convolution output channels, multi-head attention output
        and the final output of feedforward layer.
    attention_projection_dim : ``int``, required.
        The dimension of the linear projections for the self-attention layers.
    feedforward_hidden_dim : ``int``, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_blocks : ``int``, required.
        The number of stacked encoder blocks.
    num_convs_per_block: ``int``, required.
        The number of convolutions in each block.
    conv_kernel_size: ``int``, required.
        The kernel size for convolution.
    num_attention_heads : ``int``, required.
        The number of attention heads to use per layer.
    use_positional_encoding: ``bool``, optional, (default = True)
        Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
        as without this feature, the self attention layers have no idea of absolute or relative
        position (as they are just computing pairwise similarity between vectors of elements),
        which can be important features for many tasks.
    dropout_prob : ``float``, optional, (default = 0.1)
        The dropout probability for the feedforward network.
    layer_dropout_undecayed_prob : ``float``, optional, (default = 0.1)
        The initial dropout probability for layer dropout, and this might decay w.r.t the depth
        of the layer. For each mini-batch, the convolution/attention/ffn sublayer is
        stochastically dropped according to its layer dropout probability.
    attention_dropout_prob : ``float``, optional, (default = 0)
        The dropout probability for the attention distributions in the attention layer.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 attention_projection_dim: int,
                 feedforward_hidden_dim: int,
                 num_blocks: int,
                 num_convs_per_block: int,
                 conv_kernel_size: int,
                 num_attention_heads: int,
                 use_positional_encoding: bool = True,
                 dropout_prob: float = 0.1,
                 layer_dropout_undecayed_prob: float = 0.1,
                 attention_dropout_prob: float = 0) -> None:
        super().__init__()

        self._input_projection_layer = None

        if input_dim != hidden_dim:
            self._input_projection_layer = torch.nn.Linear(input_dim, hidden_dim)
        else:
            self._input_projection_layer = lambda x: x

        self._encoder_blocks: List[QaNetEncoderBlock] = []
        for block_index in range(num_blocks):
            encoder_block = QaNetEncoderBlock(hidden_dim,
                                              hidden_dim,
                                              attention_projection_dim,
                                              feedforward_hidden_dim,
                                              num_convs_per_block,
                                              conv_kernel_size,
                                              num_attention_heads,
                                              use_positional_encoding,
                                              dropout_prob,
                                              layer_dropout_undecayed_prob,
                                              attention_dropout_prob)
            self.add_module(f"encoder_block_{block_index}", encoder_block)
            self._encoder_blocks.append(encoder_block)

        self._input_dim = input_dim
        self._output_dim = hidden_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        inputs = self._input_projection_layer(inputs)
        output = inputs
        for encoder_block in self._encoder_blocks:
            output = encoder_block(output, mask)
        return output


@Seq2SeqEncoder.register("memory_efficient_multi_head_self_attention")
class MemoryEfficientMultiHeadSelfAttention(MultiHeadSelfAttention):
    # pylint: disable=line-too-long
    """
    This class is a memory efficient version of the `MultiHeadSelfAttention` in Allennlp:
    1. We divide the scale before we compute the similarity matrix.
    2. We use the `memory_effient_masked_softmax` instead of the `masked_softmax` in AllenNLP.
    """
    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """
        num_heads = self._num_heads

        batch_size, timesteps, _ = inputs.size()
        if mask is None:
            mask = inputs.new_ones(batch_size, timesteps)

        # Shape (batch_size, timesteps, 2 * attention_dim + values_dim)
        combined_projection = self._combined_projection(inputs)
        # split by attention dim - if values_dim > attention_dim, we will get more
        # than 3 elements returned. All of the rest are the values vector, so we
        # just concatenate them back together again below.
        queries, keys, *values = combined_projection.split(self._attention_dim, -1)
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = torch.cat(values, -1).contiguous()

        # Shape (num_heads * batch_size, timesteps, values_dim / num_heads)
        values_per_head = values.view(batch_size, timesteps, num_heads, int(self._values_dim / num_heads))
        values_per_head = values_per_head.transpose(1, 2).contiguous()
        values_per_head = values_per_head.view(batch_size * num_heads, timesteps, int(self._values_dim / num_heads))

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        queries_per_head = queries.view(batch_size, timesteps, num_heads, int(self._attention_dim / num_heads))
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(batch_size * num_heads, timesteps,
                                                 int(self._attention_dim / num_heads))

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        keys_per_head = keys.view(batch_size, timesteps, num_heads, int(self._attention_dim / num_heads))
        keys_per_head = keys_per_head.transpose(1, 2).contiguous()
        keys_per_head = keys_per_head.view(batch_size * num_heads, timesteps, int(self._attention_dim / num_heads))

        # shape (num_heads * batch_size, timesteps, timesteps)
        scaled_similarities = torch.bmm(queries_per_head / self._scale, keys_per_head.transpose(1, 2))

        # shape (num_heads * batch_size, timesteps, timesteps)
        # Normalise the distributions, using the same mask for all heads.
        attention = masked_softmax(scaled_similarities,
                                   mask.repeat(1, num_heads).view(batch_size * num_heads, timesteps))
        attention = self._attention_dropout(attention)

        # Take a weighted sum of the values with respect to the attention
        # distributions for each element in the num_heads * batch_size dimension.
        # shape (num_heads * batch_size, timesteps, values_dim/num_heads)
        outputs = weighted_sum(values_per_head, attention)

        # Reshape back to original shape (batch_size, timesteps, values_dim)
        # shape (batch_size, num_heads, timesteps, values_dim/num_heads)
        outputs = outputs.view(batch_size, num_heads, timesteps, int(self._values_dim / num_heads))
        # shape (batch_size, timesteps, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, timesteps, values_dim)
        outputs = outputs.view(batch_size, timesteps, self._values_dim)

        # Project back to original input size.
        # shape (batch_size, timesteps, input_size)
        outputs = self._output_projection(outputs)
        return outputs
