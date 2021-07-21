import numpy as np
import pandas as pd
import collections
import os
import functools
import json
import sys
import logging
import pickle
import re
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import time
import scipy.signal
import math

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import random 
from transformers import *
from transformers.modeling_bert import BertConfig,BertLayerNorm
from transformers.activations import gelu, gelu_new, swish

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

#bert module
def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) #(B,H,L,D)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        click_times=None,
        R=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if R is not None:
            sizes=list(query_layer.shape)
            rpr_key=torch.matmul(query_layer.permute(2,0,1,3).contiguous().view(sizes[2],-1,sizes[-1]),R.permute(0,2,1)).view(sizes[:-1]+[sizes[2]])
            attention_scores+=rpr_key
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if click_times is not None:
            attention_scores=attention_scores*click_times.unsqueeze(1).unsqueeze(1).float()
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        #(B,H,L,L) ->(B,H,L,L,1)
        #(L,L,D)  ->(1,1,L,L,D)
        if R is not None:
            rpr_value=torch.mean(attention_probs.unsqueeze(-1)*R.unsqueeze(0).unsqueeze(0),dim=-2).float()
            context_layer+=rpr_value
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        click_times=None,
        R=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask,click_times,R
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        click_times=None,
        R=None
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask,click_times=click_times,R=R)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        click_times=None,
        R=None
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask,click_times=click_times,\
                R=R
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertModel, BertTokenizer
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
def Conv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
#     return nn.Sequential(m,nn.BatchNorm1d(out_channels))
    return nn.utils.weight_norm(m)
class ResidualConv1dGLU(nn.Module):
    """Residual dilated conv1d + Gated linear unit
    Args:
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        skip_out_channels (int): Skip connection channels. If None, set to same
          as ``residual_channels``.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        dropout (float): Dropout probability.
        padding (int): Padding for convolution layers. If None, proper padding
          is computed depends on dilation and kernel_size.
        dilation (int): Dilation factor.
    """

    def __init__(self, residual_channels, gate_channels, kernel_size,
                 skip_out_channels=None,
                 cin_channels=-1, gin_channels=-1,
                 dropout=1 - 0.95, padding=None, dilation=1, causal=True,
                 bias=True, *args, **kwargs):
        super(ResidualConv1dGLU, self).__init__()
        self.dropout = dropout
        if skip_out_channels is None:
            skip_out_channels = residual_channels
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal

        self.conv = Conv1d(residual_channels, gate_channels, kernel_size,
                           padding=padding, dilation=dilation,
                           bias=bias)

        # local conditioning
        if cin_channels > 0:
            self.conv1x1c = Conv1d(cin_channels, gate_channels,kernel_size=1, bias=False)
        else:
            self.conv1x1c = None

        # global conditioning
        if gin_channels > 0:
            self.conv1x1g = Conv1d(gin_channels, gate_channels,kernel_size=1, bias=False)
        else:
            self.conv1x1g = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d(gate_out_channels, residual_channels, kernel_size=1,bias=bias)
        self.conv1x1_skip =  Conv1d(gate_out_channels, skip_out_channels, kernel_size=1,bias=bias)

    def forward(self, x, c=None, g=None):
        """Forward
        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Local conditioning features
            g (Tensor): B x C x T, Expanded global conditioning features
            is_incremental (Bool) : Whether incremental mode or not
        Returns:
            Tensor: output
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        splitdim = 1
        x = self.conv(x) ###
        # remove future time steps
        x = x[:, :, :residual.size(-1)] if self.causal else x

        a, b = x.split(x.size(splitdim) // 2, dim=splitdim) ###

        # local conditioning
        if c is not None:
            assert self.conv1x1c is not None
            c = self.conv1x1c(c)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            a, b = a + ca, b + cb

        # global conditioning
        if g is not None:
            assert self.conv1x1g is not None
            g = self.conv1x1g(g)
            ga, gb = g.split(g.size(splitdim) // 2, dim=splitdim)
            a, b = a + ga, b + gb

        x = torch.tanh(a) * torch.sigmoid(b) ###

        # For skip connection
        s = self.conv1x1_skip(x)

        # For residual connection
        x = self.conv1x1_out(x) ###

        x = (x + residual) * math.sqrt(0.5) ###
        return x, s
#辅助模块
import math
class LabelSmoothingCrossEntropy(nn.Module):


    def __init__(self, smoothing=0.1,num_class=2,weights=None):

        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        if weights is None:
            self.weights=(torch.ones(num_class)/num_class).to(device)
        else:
            self.weights=weights.to(device)

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        if target.shape==x.shape:

#             new_target=torch.zeros(X.shape).scatter_(1,target.unsqueeze(1),1)
            new_target=target
            smooth_target=new_target*self.confidence+torch.ones_like(new_target)*(self.smoothing/new_target.shape[1])
            return -(logprobs*smooth_target).sum(dim=-1).mean()
        else:
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = (-logprobs*self.weights).sum(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
def masked_softmax(X,valid_len):
    if valid_len is None:
        return F.softmax(X,dim=-1)
    else:
        shape=X.shape
        if valid_len.dim()==1:
            if X.dim()>2:
                valid_len=valid_len.view(-1,1).repeat(1,shape[1]).view(-1)
        else:
            valid_len=valid_len.view(-1)
#         print(X.shape)
        X=X.view(-1,shape[-1])

        mask=(torch.arange(0,X.shape[-1]).repeat(X.shape[0],1).to(X.device)<valid_len.view(-1,1).repeat(1,X.shape[-1])).float()
        mask=torch.log(mask)
        X=X+mask
        
        return F.softmax(X,dim=-1).view(shape)
class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(features))
        self.beta=nn.Parameter(torch.zeros(features))
        self.eps=eps
    def forward(self,X):
#         print(X.shape)
        mean=X.mean(-1,keepdim=True)
        std=X.std(-1,keepdim=True,unbiased=False)
        return self.gamma*(X-mean)/(std+self.eps)+self.beta
class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.
    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """

    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.
        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).
        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0],
                                             sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training,
                                             inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch
#attention模块
class MLPAttention(nn.Module):
    def __init__(self,query_size,key_size,units,dropout):
        super(MLPAttention,self).__init__()
        self.W_k=nn.Sequential(nn.Linear(key_size,units,bias=False)
                              ,nn.Tanh())
        self.W_q=nn.Sequential(nn.Linear(query_size,units,bias=False)
                              ,nn.Tanh())
        
        self.v=nn.Linear(units,1,bias=False)
        self.dropout=nn.Dropout(dropout)
    def forward(self,query,key,value,valid_len):
        query,key=self.W_q(query),self.W_k(key)
        features=query.unsqueeze(dim=2)+key.unsqueeze(dim=1)
        scores=self.v(features).squeeze(-1)
        attention_weights=self.dropout(masked_softmax(scores,valid_len))
        return torch.bmm(attention_weights,value)
class MLPAttention_weight(nn.Module):
    def __init__(self,query_size,key_size,units,dropout):
        super(MLPAttention_weight,self).__init__()
        self.W_k=nn.Sequential(nn.Linear(key_size,units,bias=False)
                              ,nn.Tanh())
        self.W_q=nn.Sequential(nn.Linear(query_size,units,bias=False)
                              ,nn.Tanh())
        
        self.v=nn.Linear(units,1,bias=False)
        self.dropout=nn.Dropout(dropout)
    def forward(self,query,key,value,valid_len):
        query,key=self.W_q(query),self.W_k(key)
        features=query.unsqueeze(dim=2)+key.unsqueeze(dim=1)
        scores=self.v(features).squeeze(-1)
        attention_weights=self.dropout(masked_softmax(scores,valid_len))
        return attention_weights
    
class AddictiveAttention(nn.Module):
    def __init__(self,key_size,dropout=0):
        super(AddictiveAttention,self).__init__()
        self.W_k=nn.Sequential(nn.Linear(key_size,1,bias=False)
                              ,nn.Tanh())
        
#         self.v=nn.Linear(units,1,bias=False)
        self.dropout=nn.Dropout(dropout)
    def forward(self,key,value,valid_len=None):
        key=self.W_k(key)
        scores=key.squeeze(-1)
        attention_weights=self.dropout(masked_softmax(scores,valid_len)).unsqueeze(-1)
        return (attention_weights*value).sum(dim=1)
class R_Attention(nn.Module):
    def __init__(self,query_size,key_size,dropout=0):
        super(R_Attention,self).__init__()
        self.rnn=nn.GRUCell(key_size,query_size)
        self.attention=MLPAttention(query_size,key_size,max(query_size,key_size),dropout)
    def forward(self,sequence,N=3,valid_len=None):
        h=torch.zeros(sequence.shape[0],self.rnn.hidden_size).to(sequence.device)
        if N==1:
            inputs=self.attention(h.unsqueeze(1),sequence,sequence,valid_len).squeeze(1)
            return inputs
        for i in range(N):
            inputs=self.attention(h.unsqueeze(1),sequence,sequence,valid_len).squeeze(1)
            h=self.rnn(inputs,h)
        return h
    
class DotProductAttention(nn.Module):
    def __init__(self,dropout=0):
        super(DotProductAttention,self).__init__()
        self.dropout=nn.Dropout(dropout)
    # query: (batch_size, #queries, d)
    # key: (batch_size, #kv_pairs, d)
    # value: (batch_size, #kv_pairs, dim_v)
    # valid_len: either (batch_size, ) or (batch_size, xx)
    def forward(self,query,key,value,valid_len=None):
        d=query.shape[-1]
        shape=query.shape
        if valid_len is None:
            valid_len=torch.ones(key.shape[0]).long().to(key.device)
        if  valid_len.dim()==1:
            valid_len=valid_len.view(-1,1).repeat(1,shape[1])
        mask=(torch.arange(0,query.shape[1]).repeat(query.shape[0],1).to(query.device)<valid_len).float()
        scores=torch.bmm(query,key.permute(0,2,1))/math.sqrt(d)
        attention_weights=self.dropout(masked_softmax(scores,valid_len))
        return torch.bmm(attention_weights,value)*mask.unsqueeze(-1)
class DotProductAttention_weight(nn.Module):
    def __init__(self,query_size,key_size,dropout=0):
        super(DotProductAttention_weight,self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.W_q=nn.Linear(query_size,key_size,bias=False)
    def forward(self,query,key,value,valid_len=None):
        query=self.W_q(query)
        d=key.shape[-1]
        shape=query.shape
        if valid_len is None:
            valid_len=torch.ones(key.shape[0]).long().to(key.device)
        if  valid_len.dim()==1:
            valid_len=valid_len.view(-1,1).repeat(1,shape[1])
        mask=(torch.arange(0,query.shape[1]).repeat(query.shape[0],1).to(query.device)<valid_len).float()
        scores=torch.bmm(query,key.permute(0,2,1))/math.sqrt(d)
        attention_weights=self.dropout(masked_softmax(scores,valid_len))
        return attention_weights
    
class AttentionSequencePoolingLayer(nn.Module):
    """The Attentional sequence pooling operation used in DIN & DIEN.
        Arguments
          - **att_hidden_units**:list of positive integer, the attention net layer number and units in each layer.
          - **att_activation**: Activation function to use in attention net.
          - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.
          - **supports_masking**:If True,the input need to support masking.
        References
          - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
      """

    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False,
                 return_score=False, supports_masking=False, embedding_dim=4, **kwargs):
        super(AttentionSequencePoolingLayer, self).__init__()
        self.return_score = return_score
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking
        self.local_att = LocalActivationUnit(hidden_units=att_hidden_units, embedding_dim=embedding_dim,
                                             activation=att_activation,
                                             dropout_rate=0, use_bn=False)

    def forward(self, query, keys, keys_length, mask=None):
        """
        Input shape
          - A list of three tensor: [query,keys,keys_length]
          - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``
          - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``
          - keys_length is a 2D tensor with shape: ``(batch_size, 1)``
        Output shape
          - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
        """
        batch_size, max_length, dim = keys.size()

        # Mask
        if self.supports_masking:
            if mask is None:
                raise ValueError("When supports_masking=True,input must support masking")
            keys_masks = mask.unsqueeze(1)
        else:
            keys_masks = torch.arange(max_length, device=keys_length.device, dtype=keys_length.dtype).repeat(batch_size,
                                                                                                             1)  # [B, T]
            keys_masks = keys_masks < keys_length.view(-1, 1)  # 0, 1 mask
            keys_masks = keys_masks.unsqueeze(1)  # [B, 1, T]

        attention_score = self.local_att(query, keys)  # [B, T, 1]

        outputs = torch.transpose(attention_score, 1, 2)  # [B, 1, T]

        if self.weight_normalization:
            paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = torch.zeros_like(outputs)

        outputs = torch.where(keys_masks, outputs, paddings)  # [B, 1, T]

        # Scale
        # outputs = outputs / (keys.shape[-1] ** 0.05)

        if self.weight_normalization:
            outputs = F.softmax(outputs, dim=-1)  # [B, 1, T]

        if not self.return_score:
            # Weighted sum
            outputs = torch.matmul(outputs, keys)  # [B, 1, E]

        return outputs
#sequence model
class KMaxPooling(nn.Module):
    """K Max pooling that selects the k biggest value along the specific axis.
      Input shape
        -  nD tensor with shape: ``(batch_size, ..., input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., output_dim)``.
      Arguments
        - **k**: positive integer, number of top elements to look for along the ``axis`` dimension.
        - **axis**: positive integer, the dimension to look for elements.
     """

    def __init__(self, k, axis, device='cpu'):
        super(KMaxPooling, self).__init__()
        self.k = k
        self.axis = axis
        self.to(device)

    def forward(self, input):
        if self.axis < 0 or self.axis >= len(input.shape):
            raise ValueError("axis must be 0~%d,now is %d" %
                             (len(input.shape) - 1, self.axis))

        if self.k < 1 or self.k > input.shape[self.axis]:
            raise ValueError("k must be in 1 ~ %d,now k is %d" %
                             (input.shape[self.axis], self.k))

        out = torch.topk(input, k=self.k, dim=self.axis, sorted=True)[0]
        return out
class AGRUCell(nn.Module):
    """ Attention based GRU (AGRU)
        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.normal(torch.zeros(3 * hidden_size, input_size),std=0.01))
        self.register_parameter('weight_ih', self.weight_ih)
        for i in range(3):
            nn.init.xavier_uniform_(self.weight_ih[hidden_size*i:hidden_size*(i+1)])
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.normal(torch.zeros(3 * hidden_size, hidden_size),std=0.01))
        for i in range(3):
            nn.init.orthogonal_(self.weight_hh[hidden_size*i:hidden_size*(i+1)])
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_hh', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, input, hx, att_score):
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        # update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        hy = (1. - att_score) * hx + att_score * new_state
        return hy

class AUGRUCell(nn.Module):
    """ Effect of GRU with attentional update gate (AUGRU)
        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.normal(torch.zeros(3 * hidden_size, input_size),std=0.01))
        self.register_parameter('weight_ih', self.weight_ih)
        for i in range(3):
            nn.init.xavier_uniform_(self.weight_ih[hidden_size*i:hidden_size*(i+1)])
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.normal(torch.zeros(3 * hidden_size, hidden_size),std=0.01))
        for i in range(3):
            nn.init.orthogonal_(self.weight_hh[hidden_size*i:hidden_size*(i+1)])
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, input, hx, att_score):
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)
    
        att_score = att_score.view(-1, 1)
        update_gate = att_score * update_gate
        hy = (1. - update_gate) * hx + update_gate * new_state
        return hy

class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, gru_type='AGRU'):
        super(DynamicGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if gru_type == 'AGRU':
            self.rnn = AGRUCell(input_size, hidden_size, bias)
        elif gru_type == 'AUGRU':
            self.rnn = AUGRUCell(input_size, hidden_size, bias)

    def forward(self, input, att_scores=None, hx=None):
        if not isinstance(input, PackedSequence) or not isinstance(att_scores, PackedSequence):
            raise NotImplementedError("DynamicGRU only supports packed input and att_scores")

        input, batch_sizes, sorted_indices, unsorted_indices = input
        att_scores, _, _, _ = att_scores

        max_batch_size = int(batch_sizes[0])
        if hx is None:
            hx = torch.zeros(max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)

        outputs = torch.zeros(input.size(0), self.hidden_size,
                              dtype=input.dtype, device=input.device)

        begin = 0
        for idx,batch in enumerate(batch_sizes):
            new_hx = self.rnn(
                input[begin:begin + batch],
                hx[0:batch],
                att_scores[begin:begin + batch])
            outputs[begin:begin + batch] = new_hx
            hx = new_hx
            begin += batch
        return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)
class AUGRU_composition(nn.Module):
    #attention 得替换 mlp还是好一些
    #AUGRU/AIGRU/AGRU  
    def __init__(self,input_size,hidden_size,query_size,attention_type='dot',**kwargs):
        super(AUGRU_composition,self).__init__(**kwargs)
        self.net=DynamicGRU(input_size,hidden_size,gru_type='AUGRU')

        if attention_type=='dot':
            self.attunit=DotProductAttention_weight(query_size,input_size,0)
        else:
            self.attunit=MLPAttention_weight(query_size,input_size,input_size*4,0)
    def forward(self,sequence,query):
        query=query.unsqueeze(dim=1)
        att_scores=self.attunit(query,sequence,sequence,None).squeeze(dim=1)
        sequence=pack_padded_sequence(sequence,lengths=torch.ones(sequence.shape[0])*sequence.shape[1],batch_first=True)
        att_scores=pack_padded_sequence(att_scores,lengths=torch.ones(att_scores.shape[0])*att_scores.shape[1],batch_first=True)
        outputs=pad_packed_sequence(self.net(sequence,att_scores),batch_first=True)[0]
        return outputs

class TextCNN(nn.Module):
    def __init__(self,input_size,kernel_sizes,num_channels):
        super(TextCNN,self).__init__()
        self.convs1=nn.ModuleList()
        self.convs2=nn.ModuleList()
        self.conv3=nn.Sequential(nn.Conv1d(in_channels=sum(num_channels),out_channels=sum(num_channels),\
                                           kernel_size=kernel_sizes[0],padding=1)
                                            ,nn.BatchNorm1d(sum(num_channels)))
        for i,(k,c) in enumerate(zip(kernel_sizes,num_channels)):
            self.convs1.append(nn.Sequential(nn.Conv1d(in_channels=input_size,out_channels=c,kernel_size=k,padding=i+1)
                                            ,nn.BatchNorm1d(c)))
            if i<len(kernel_sizes)-1:
                self.convs2.append(nn.Sequential(nn.Conv1d(in_channels=sum(num_channels),\
                                                out_channels=sum(num_channels)//(len(kernel_sizes)-1),\
                                                kernel_size=k,padding=i+1),nn.BatchNorm1d(sum(num_channels)//(len(kernel_sizes)-1))))
    def forward(self,X):#(batch,seq_len)
        outputs=torch.cat([F.max_pool1d(F.relu(conv(X)),2) for conv in self.convs1],dim=1)
        outputs=torch.cat([F.max_pool1d(F.relu(conv(outputs)),2,padding=1) for conv in self.convs2],dim=1)
        outputs=F.max_pool1d(F.relu(self.conv3(outputs)),2)
        return outputs
class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        max_len=config.max_position_embeddings
        self.P=torch.zeros(1,max_len,config.hidden_size)
        X=torch.arange(0,max_len).view(-1,1).float()/torch.pow(10000,torch.arange(0,config.hidden_size,2).float()/config.hidden_size)
        self.P[:,:,0::2]=torch.sin(X.float())
        self.P[:,:,1::2]=torch.cos(X.float())
        self.Dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    def forward(self,X,position_ids=None):
        input_shape = X.size()[:-1]
        seq_length = input_shape[1]
        device = X.device
        inputs_embeds=X
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings

        embeddings = self.layerNorm(embeddings)

        embeddings = self.Dropout(embeddings)
        return embeddings
class XY_Encoding(nn.Module):
    def __init__(self, config,sinusoidal=False,use_layernorm=True):
        super(XY_Encoding, self).__init__()
        self.sinusoidal = sinusoidal
        self.position_embeddings_x = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.position_embeddings_y = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        max_len=config.max_position_embeddings
        self.P=torch.zeros(max_len,config.hidden_size)
        X=torch.arange(0,max_len).view(-1,1).float()/torch.pow(10000,torch.arange(0,config.hidden_size,2).float()/config.hidden_size)
        self.P[:,0::2]=torch.sin(X.float())
        self.P[:,1::2]=torch.cos(X.float())
        self.res = nn.Embedding(max_len,config.hidden_size,_weight = self.P)
        self.res.weight.requires_grad = False
        self.Dropout = nn.Dropout(config.hidden_dropout_prob)
        self.use_layernorm = use_layernorm
        self.layerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    def forward(self,X,position_ids = None):
        x,y = None,None
        if position_ids is not None:
            x = position_ids[:,:,0]
            y = position_ids[:,:,1]
        input_shape = X.size()[:-1]
        seq_length = input_shape[1]
        device = X.device
        inputs_embeds=X
        if x is None:
            x = torch.arange(seq_length, dtype=torch.long, device=device)
            x = x.unsqueeze(0).expand(input_shape)
        if y is None:
            y = torch.arange(seq_length, dtype=torch.long, device=device)
            y = y.unsqueeze(0).expand(input_shape)
        if not self.sinusoidal:
            x = self.position_embeddings_x(x)
            y = self.position_embeddings_y(y)
        else:
            x = self.res(x)
            y = self.res(y)

        embeddings = inputs_embeds + x + y
        if self.use_layernorm:
            embeddings = self.layerNorm(embeddings)

        embeddings = self.Dropout(embeddings)
        return embeddings
class Transformer_Encoder(nn.Module):
    def __init__(self,hidden_size=128,num_hidden_layers=4,num_attention_heads=4,max_position_embeddings=8000):
        super(Transformer_Encoder,self).__init__()
        config = BertConfig(vocab_size=1000,hidden_size=hidden_size,intermediate_size = hidden_size*4,num_hidden_layers=num_hidden_layers,\
                            num_attention_heads=num_attention_heads,max_position_embeddings=max_position_embeddings,\
                            output_attentions=True,output_hidden_states=True)
        self.config = config
        self.Encoder = BertEncoder(config=config)
        self.P = PositionalEncoding(config)
        self.pooler = AddictiveAttention(config.hidden_size,dropout=config.hidden_dropout_prob)
        for n,e in self.Encoder.named_modules():
            self._init_weights(e)
        for n,e in self.P.named_modules():
            self._init_weights(e)
    def replace_masked(self,tensor, mask, value):
        mask = mask.unsqueeze(1).transpose(2, 1)
        reverse_mask = 1.0 - mask
        values_to_add = value * reverse_mask
        return tensor * mask + values_to_add
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):

            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    def make_mask(self,X,valid_len):
            shape=X.shape[:2]
            if valid_len.dim() == 1:
                valid_len = valid_len.view(-1,1).repeat(1,shape[1])
            mask=(torch.arange(0,X.shape[1]).repeat(X.shape[0],1).to(X.device)<valid_len).float()
            return mask
    def forward(self,X,length,position_encode = True,position_ids = None,decode = False,query_state = None):

        attention_mask = self.make_mask(X,length)


        if position_encode:
            embedding_output=self.P(X,position_ids)
        else:
            embedding_output = X

        #adjust attention mask
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        if decode:
            extended_attention_mask  =  extended_attention_mask.repeat(1,1, extended_attention_mask.shape[-1],1)
            self_mask = torch.arange(X.shape[1]).repeat(X.shape[1],1)
            self_mask = (self_mask<=torch.arange(X.shape[1]).unsqueeze(1)).float().to(X.device)
            extended_attention_mask = extended_attention_mask * self_mask.unsqueeze(0).unsqueeze(0)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        #make head mask
        head_mask = [None] * self.config.num_hidden_layers
        outputs=self.Encoder(  embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None)
        X=self.pooler(outputs[0],outputs[0],length)
        return X,outputs[0]

class GRU_module_AUGRU(nn.Module):
    def __init__(self,input_size,hidden_size,feature_num,dropout_rate,bidirectional=False,model_type='regression'):
        super(GRU_module_AUGRU,self).__init__()
        self.dropout=dropout_rate
        LayerNorm=nn.LayerNorm
        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            self.dense_dropout=nn.Dropout(self.dropout)
        seis_size=6 if use_V else 3
        output_size=input_size if input_size<=seis_size else seis_size
        input_size_fre=input_size
        self.cnn=TextCNN(input_size,[3,5,7],[5,6,5])
#         nn.Sequential(nn.Conv1d(input_size,16,3,padding=1),
#                           nn.BatchNorm1d(16),
#                           nn.ReLU(),
#                           nn.MaxPool1d(2,padding=0),
#                           nn.Conv1d(16,16,3,padding=1),
#                           nn.BatchNorm1d(16),
#                           nn.ReLU(),
#                           nn.MaxPool1d(2,padding=1),
#                           nn.Conv1d(16,16,3,padding=1),
#                           nn.BatchNorm1d(16),
#                           nn.ReLU(),
#                           nn.MaxPool1d(2,padding=0))
        self.reverse_cnn=nn.Sequential(nn.ConvTranspose1d(in_channels=16,out_channels=16,kernel_size=3,stride=2,padding=1),
                            nn.BatchNorm1d(num_features=16),
                            nn.ReLU(True),
                            nn.ConvTranspose1d(in_channels=16,out_channels=16,kernel_size=3,stride=2,padding=1,output_padding=1),
                            nn.BatchNorm1d(num_features=16),
                            nn.ReLU(True),
                            nn.ConvTranspose1d(in_channels=16,out_channels=output_size,kernel_size=3,stride=2,padding=1,output_padding=1))

        self.rnn=AUGRU_composition(16,hidden_size,query_size=hidden_size,attention_type='mlp')
        #nn.GRU(input_size=16,hidden_size=hidden_size,batch_first=True,bidirectional=False)
        #
        self.rnn_att=R_Attention(hidden_size,hidden_size)

        if use_phase:
            input_size_fre*=2
        self.cnn_fre=nn.Sequential(nn.Conv1d(input_size_fre,hidden_size,3,padding=1),
                          nn.BatchNorm1d(hidden_size),
                          nn.ReLU(),
                          nn.MaxPool1d(2,padding=0),
                          nn.Conv1d(hidden_size,hidden_size,3,padding=1),
                          nn.BatchNorm1d(hidden_size),
                          nn.ReLU(),
                          nn.MaxPool1d(2,padding=1))
#         self.cnn_fre=TextCNN(input_size_fre,[3,5,7],[hidden_size//3,hidden_size//3,hidden_size-hidden_size//3*2])
        self.rnn_fre=nn.GRU(input_size=input_size_fre,hidden_size=hidden_size,batch_first=True,bidirectional=bidirectional)
        self.rnn_fre_att=R_Attention(hidden_size,hidden_size)
        self.fre_trans=nn.Linear(input_size_fre*51,hidden_size,bias=False)
#         self.attunit=MLPAttention_weight(hidden_size,16,16*4,0)
        self.fre_att=AddictiveAttention(hidden_size)

        self._attention =DotProductAttention()

        self.hidden_size=hidden_size*(2 if self.rnn_fre.bidirectional else 1)
        if self.rnn_fre.bidirectional:
            self.dense1=nn.Linear(2*self.hidden_size+hidden_size,self.hidden_size)
        else:
            self.dense1=nn.Linear(self.hidden_size+hidden_size,self.hidden_size)
        if model_type=='regression':
            self.dense2=nn.Linear(self.hidden_size,1)
        else:
            self.dense2=nn.Linear(self.hidden_size,5)
            self.reg_dense2=nn.Linear(self.hidden_size,1)
        self.model_type=model_type
        self.ln_rnn=LayerNorm(hidden_size)
        self.ln_fre=LayerNorm(self.hidden_size)
        self.ln_feature=LayerNorm(feature_num)
    def forward(self,X,X_fre,**kwargs):
#         X=self.ln(X)

        reg_output=None
        X=self.cnn(X.permute(0,2,1)).permute(0,2,1)   #CNN
        new_X=self.reverse_cnn(X.permute(0,2,1)).permute(0,2,1)

        if self.dropout:
            X=self._rnn_dropout(X)
#             X_fre=self._rnn_dropout(X_fre)
        X_fre=self.dense_dropout(X_fre)

#         Y_fre,_=self.rnn_fre(X_fre) #RNN
#         Y_fre_hidden=Y_fre[:,-1,:]
#         Y_fre_hidden=self.fre_att(Y_fre,Y_fre).sum(dim=1)
#         Y_fre_hidden=F.adaptive_avg_pool1d(Y_fre.permute(0,2,1),1).squeeze(-1)
#         Y_fre_hidden=torch.cat([F.adaptive_avg_pool1d(Y_fre.permute(0,2,1),1).squeeze(-1),\
#                                 F.adaptive_max_pool1d(Y_fre.permute(0,2,1),1).squeeze(-1)],dim=1)
#         Y_fre=self.ln_fre(Y_fre)
    
        Y_fre=F.adaptive_avg_pool1d(self.cnn_fre(X_fre.permute(0,2,1)),1).permute(0,2,1) #CNN
        Y_fre_hidden=Y_fre[:,-1]

#         Y_fre_output=self.rnn_fre_att(Y_fre,N=1)
        
#频域encoder的三种选择 GRU/TCN/Conv-LSTM 

#         att_scores=self.attunit(Y_fre_hidden.unsqueeze(dim=1),X,X,None).squeeze(dim=1)
#         X*=att_scores.unsqueeze(dim=-1)
#         Y_time,_=self.rnn(X)  #RNN

#         Y_fre_hidden=self.fre_trans(X_fre.view(X_fre.shape[0],-1))

        Y_time=self.rnn(X,Y_fre_hidden)
    
        Y_time=self.ln_rnn(Y_time)
#         Y_time_output=self.rnn_att(Y_time,N=3)

#时域encoder的三种选择 GRU/TCN/Conv-LSTM 

        
        if self.rnn_fre.bidirectional:   #这个地方可以考虑把首尾的一半concat起来，就不要那个妹有时间的部分
            inputs=torch.cat([Y_time[:,-1,:],Y_fre[:,-0,:],Y_fre[:,-1,:]],dim=-1)  #前后cat
        else:
            inputs=torch.cat([Y_time[:,-1,:],Y_fre[:,-1,:]],dim=-1)  #前后cat
        output=self.dense1(self.dense_dropout(inputs))
        if self.model_type!='regression':
            reg_output=self.reg_dense2(output).squeeze(-1)
        output=self.dense2(output)
        if self.model_type=='regression':
            output=output.squeeze(-1)

        return output,new_X,reg_output
    def get_hidden(self,X,X_fre,**kwargs):
#         X=self.ln(X)

        reg_output=None
        X=self.cnn(X.permute(0,2,1)).permute(0,2,1)   #CNN
        new_X=self.reverse_cnn(X.permute(0,2,1)).permute(0,2,1)

        if self.dropout:
            X=self._rnn_dropout(X)
#             X_fre=self._rnn_dropout(X_fre)
        X_fre=self.dense_dropout(X_fre)

#         Y_fre,_=self.rnn_fre(X_fre) #RNN
#         Y_fre_hidden=Y_fre[:,-1,:]
#         Y_fre_hidden=F.adaptive_avg_pool1d(Y_fre.permute(0,2,1),1).squeeze(-1)
#         Y_fre_hidden=torch.cat([F.adaptive_avg_pool1d(Y_fre.permute(0,2,1),1).squeeze(-1),\
#                                 F.adaptive_max_pool1d(Y_fre.permute(0,2,1),1).squeeze(-1)],dim=1)
#         Y_fre_hidden=self.fre_att(Y_fre,Y_fre).sum(dim=1)
#         Y_fre=self.ln_fre(Y_fre)
    
        Y_fre=F.adaptive_avg_pool1d(self.cnn_fre(X_fre.permute(0,2,1)),1).permute(0,2,1) #CNN
        Y_fre_hidden=Y_fre[:,-1]

#         Y_fre_output=self.rnn_fre_att(Y_fre,N=1)
        
#频域encoder的三种选择 GRU/TCN/Conv-LSTM 

#         att_scores=self.attunit(Y_fre_hidden.unsqueeze(dim=1),X,X,None).squeeze(dim=1)
#         X*=att_scores.unsqueeze(dim=-1)
#         Y_time,_=self.rnn(X)  #RNN

#         Y_fre_hidden=self.fre_trans(X_fre.view(X_fre.shape[0],-1))
        Y_time=self.rnn(X,Y_fre_hidden)
    
        Y_time=self.ln_rnn(Y_time)
#         Y_time_output=self.rnn_att(Y_time,N=3)

#时域encoder的三种选择 GRU/TCN/Conv-LSTM 

        
        if self.rnn_fre.bidirectional:   #这个地方可以考虑把首尾的一半concat起来，就不要那个妹有时间的部分
            inputs=torch.cat([Y_time[:,-1,:],Y_fre[:,-0,:],Y_fre[:,-1,:]],dim=-1)  #前后cat
        else:
            inputs=torch.cat([Y_time[:,-1,:],Y_fre[:,-1,:]],dim=-1)  #前后cat
        output=self.dense1(self.dense_dropout(inputs))
        return output
class Seis_transformer(nn.Module):
    def __init__(self,config_fre,config_time,input_size,device,model_type='regression'):
        super(Seis_transformer,self).__init__()
        self.config_fre=config_fre
        self.config_time=config_time
        seis_size=6 if use_V else 3
        output_size=input_size if input_size<=seis_size else seis_size
        input_size_fre=input_size
        output_size_fre=output_size
        if use_phase:
            input_size_fre*=2
        if use_phase:
            output_size_fre*=2
        self.cnn_fre=nn.Sequential(nn.Conv1d(input_size_fre,config.hidden_size,3,padding=1),
                          nn.BatchNorm1d(config.hidden_size),
                          nn.ReLU(),
                          nn.MaxPool1d(2,padding=0),
                          nn.Conv1d(config.hidden_size,config.hidden_size,3,padding=1),
                          nn.BatchNorm1d(config.hidden_size),
                          nn.ReLU(),
                          nn.MaxPool1d(2,padding=1))
        self.reverse_cnn_fre=nn.Sequential(nn.ConvTranspose1d(in_channels=config.hidden_size,out_channels=config.hidden_size,kernel_size=3,stride=2,padding=1,output_padding=1),
                            nn.BatchNorm1d(num_features=config.hidden_size),
                            nn.ReLU(True),
                            nn.ConvTranspose1d(in_channels=config.hidden_size,out_channels=output_size_fre,kernel_size=3,stride=2,padding=1))
#                           nn.Conv1d(config.hidden_size,config.hidden_size,3,padding=1),
#                           nn.BatchNorm1d(config.hidden_size),
#                           nn.ReLU(),
#                           nn.MaxPool1d(2,padding=0)
        self.cnn_time=nn.Sequential(nn.Conv1d(input_size,config_time.hidden_size,3,padding=1),
                          nn.BatchNorm1d(config_time.hidden_size),
                          nn.ReLU(),
                          nn.MaxPool1d(2,padding=0),
                          nn.Conv1d(config_time.hidden_size,config_time.hidden_size,3,padding=1),
                          nn.BatchNorm1d(config_time.hidden_size),
                          nn.ReLU(),
                          nn.MaxPool1d(2,padding=1),
                          nn.Conv1d(config_time.hidden_size,config_time.hidden_size,3,padding=1),
                          nn.BatchNorm1d(config_time.hidden_size),
                          nn.ReLU(),
                          nn.MaxPool1d(2,padding=0))
        self.reverse_cnn_time=nn.Sequential(nn.ConvTranspose1d(in_channels=config_time.hidden_size,out_channels=config_time.hidden_size,kernel_size=3,stride=2,padding=1),
                            nn.BatchNorm1d(num_features=config_time.hidden_size),
                            nn.ReLU(True),
                            nn.ConvTranspose1d(in_channels=config_time.hidden_size,out_channels=config_time.hidden_size,kernel_size=3,stride=2,padding=1,output_padding=1),
                            nn.BatchNorm1d(num_features=config_time.hidden_size),
                            nn.ReLU(True),
                            nn.ConvTranspose1d(in_channels=config_time.hidden_size,out_channels=output_size,kernel_size=3,stride=2,padding=1,output_padding=1))
#         self.time_embedding_x=nn.Embedding.from_pretrained(torch.tensor(wv_embeddings[0]).float(),freeze=True)
#         self.time_embedding_y=nn.Embedding.from_pretrained(torch.tensor(wv_embeddings[1]).float(),freeze=True)
#         self.time_embedding_z=nn.Embedding.from_pretrained(torch.tensor(wv_embeddings[2]).float(),freeze=True)
#         self.time_embedding_a=nn.Embedding.from_pretrained(torch.tensor(wv_embeddings[3]).float(),freeze=True)
#         k=2
#         self.rpr_embedding=nn.Embedding(2*k+1,config.hidden_size//config.num_attention_heads)
#         self.R_idx=torch.zeros(config.max_position_embeddings,config.max_position_embeddings).to(device)
#         for i in range(self.R_idx.shape[0]):
#             for j in range(self.R_idx.shape[1]):
#                 self.R_idx[i][j]=max(-k, min(k, j-i))+k
#         self.R_idx=self.R_idx.long()

        self.rnn_fre=nn.GRU(input_size=config.hidden_size,hidden_size=config.hidden_size,batch_first=True,bidirectional=False)
        self.rnn_fre_att=R_Attention(config.hidden_size,config.hidden_size)
        #TemporalConvNet(16,[16,32,32])
        #ConvLSTM(1,1,(3,),1,True)

        self.rnn_time=nn.GRU(input_size=config_time.hidden_size,hidden_size=config_time.hidden_size,batch_first=True,bidirectional=False)
        self.rnn_time_att=R_Attention(config_time.hidden_size,config_time.hidden_size)
        #TemporalConvNet(3,[16,32,32])
        
        self.Encoder_fre=BertEncoder(config=config_fre)
#         self.Encoder_fre=BertAttention(config=config)

        self.P_fre=PositionalEncoding(config_fre)
        for n,e in self.Encoder_fre.named_modules():
            self._init_weights(e)
        for n,e in self.P_fre.named_modules():
            self._init_weights(e)

        self.Encoder_time=BertEncoder(config=config_time)
#         self.Encoder_time=BertAttention(config=config_time)
        self.P_time=PositionalEncoding(config_time)
        for n,e in self.Encoder_time.named_modules():
            self._init_weights(e)
        for n,e in self.P_time.named_modules():
            self._init_weights(e)
        
        
#         self.rnns=nn.ModuleList()
#         self.device=device
#         for i in range(1):
#             hidden_size=config.hidden_size
#             self.rnns.append(nn.GRU(input_size=config.hidden_size,hidden_size=config.hidden_size,num_layers=1,\
#                         bidirectional=True,batch_first=True))
#         self.one=nn.Parameter(torch.tensor([1]).float())
#         self.pooler=nn.MaxPool1d(4)
#         self.decoder=nn.Sequential(nn.Linear(config.hidden_size*4,config.hidden_size//2),nn.Tanh(),
#                                    nn.Linear(config.hidden_size//2,1))
        if model_type=='regression':
            self.decoder=nn.Sequential(nn.Linear(2*config.hidden_size+3*config_time.hidden_size,1))
        else:
            self.decoder=nn.Sequential(nn.Linear(2*config.hidden_size+3*config_time.hidden_size),5)
            self.reg_decoder=nn.Sequential(nn.Linear(2*config.hidden_size+3*config_time.hidden_size),1)
        self.model_type=model_type
    def replace_masked(self,tensor, mask, value):
        mask = mask.unsqueeze(1).transpose(2, 1)
        reverse_mask = 1.0 - mask
        values_to_add = value * reverse_mask
        return tensor * mask + values_to_add
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):

            module.weight.data.normal_(mean=0.0, std=self.config_fre.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    def make_mask(self,X,valid_len):
            shape=X.shape
            if valid_len.dim()==1:
                valid_len=valid_len.view(-1,1).repeat(1,shape[1])
            mask=(torch.arange(0,X.shape[1]).repeat(X.shape[0],1).to(X.device)<valid_len).float()
            return mask
    def forward(self,X,X_fre,encoder_hidden_states=None,encoder_extended_attention_mask=None,**kwargs):
#         X=X.long()
#         x,y,z,a=self.time_embedding_x(X[:,:,0]),self.time_embedding_y(X[:,:,1]),self.time_embedding_z(X[:,:,2]),\
#                 self.time_embedding_z(X[:,:,3]) 
#         X=torch.cat([x,y,z],dim=-1)   #embedding
        X_time=X
        X_time=self.cnn_time(X_time.permute(0,2,1)).permute(0,2,1)   #CNN
        new_X_time=self.reverse_cnn_time(X_time.permute(0,2,1)).permute(0,2,1)
#         X_time,_=self.rnn_time(X_time)
        length=(torch.ones(X_time.shape[0])*X_time.shape[1]).to(X_time.device).long()
        #make attention mask
        attention_mask=self.make_mask(X_time,length)
#         R=self.rpr_embedding(self.R_idx[:X_time.shape[1],:X_time.shape[1]])
        embedding_output=self.P_time(X_time)

        #adjust attention mask
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        #make head mask
        head_mask = [None] * self.config_time.num_hidden_layers
        outputs=self.Encoder_time(  embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask)
        X_time=outputs[0]
        time_output=self.rnn_time_att(X_time)
    
    
#         X=X_fre
        X_fre=self.cnn_fre(X_fre.permute(0,2,1)).permute(0,2,1)   #CNN

        new_X_fre=self.reverse_cnn_fre(X_fre.permute(0,2,1)).permute(0,2,1)
#         X_fre,_=self.rnn(X_fre)
        length_fre=(torch.ones(X_fre.shape[0])*X_fre.shape[1]).to(X_fre.device).long()
        #make attention mask
        attention_mask=self.make_mask(X_fre,length_fre)
#         R=self.rpr_embedding(self.R_idx[:X_fre.shape[1],:X_fre.shape[1]])
        embedding_output=self.P_fre(X_fre)

        #adjust attention mask
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        #make head mask
        head_mask = [None] * self.config_fre.num_hidden_layers
        outputs_fre=self.Encoder_fre(  embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask)
        X_fre=outputs_fre[0]
#         fre_output=self.rnn_att_fre(X_fre)
#         X_fre=self.pooler(X_fre.permute(0,2,1)).permute(0,2,1)
#         attention_mask=self.make_mask(X_fre,length//4)
#         begin_hiddens,end_hiddens=[],[]
#         for i in range(1):
#             total_length=X_fre.shape[1]
#             new_X_fre=pack_padded_sequence(X_fre,length//4,batch_first=True,enforce_sorted=False)
#             hidden,_=self.rnns[i](new_X_fre)
#             hidden,_=pad_packed_sequence(hidden,batch_first=True)
#             hidden_avg = torch.sum(hidden * attention_mask.unsqueeze(1).transpose(2, 1), dim=1)\
#                                          / torch.sum(attention_mask, dim=1, keepdim=True)

#             hidden_max, _ = self.replace_masked(hidden, attention_mask, -1e7).max(dim=1)
#             begin_hiddens.append(hidden_avg)
#             end_hiddens.append(hidden_max)

#         begin_hidden=torch.cat(begin_hiddens,dim=-1)
#         end_hidden=torch.cat(end_hiddens,dim=-1)
#         hidden=torch.cat([begin_hidden,end_hidden],dim=-1)
        hidden=torch.cat([F.adaptive_avg_pool1d(X_fre.permute(0,2,1),1).squeeze(-1),\
                          F.adaptive_avg_pool1d(X_time.permute(0,2,1),1).squeeze(-1),\
                          F.adaptive_max_pool1d(X_fre.permute(0,2,1),1).squeeze(-1),\
                          F.adaptive_max_pool1d(X_time.permute(0,2,1),1).squeeze(-1),\
                            time_output],dim=-1)
        reg_output=None
        if self.model_type=='regression':
            return self.decoder(hidden).squeeze(-1),new_X_time,new_X_fre,reg_output
        else:
            return self.decoder(hidden),new_X_time,new_X_fre,self.reg_decoder(hidden).squeeze(-1)          
    def get_hidden(self,X,X_fre,encoder_hidden_states=None,encoder_extended_attention_mask=None,**kwargs):
#         X=X.long()
#         x,y,z,a=self.time_embedding_x(X[:,:,0]),self.time_embedding_y(X[:,:,1]),self.time_embedding_z(X[:,:,2]),\
#                 self.time_embedding_z(X[:,:,3]) 
#         X=torch.cat([x,y,z],dim=-1)   #embedding
        X_time=X
        X_time=self.cnn_time(X_time.permute(0,2,1)).permute(0,2,1)   #CNN
        new_X_time=self.reverse_cnn_time(X_time.permute(0,2,1)).permute(0,2,1)
#         X_time,_=self.rnn_time(X_time)
        length=(torch.ones(X_time.shape[0])*X_time.shape[1]).to(X_time.device).long()
        #make attention mask
        attention_mask=self.make_mask(X_time,length)
#         R=self.rpr_embedding(self.R_idx[:X_time.shape[1],:X_time.shape[1]])
        embedding_output=self.P_time(X_time)

        #adjust attention mask
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        #make head mask
        head_mask = [None] * self.config_time.num_hidden_layers
        outputs=self.Encoder_time(  embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask)
        X_time=outputs[0]
        time_output=self.rnn_time_att(X_time)
    
    
#         X=X_fre
        X_fre=self.cnn_fre(X_fre.permute(0,2,1)).permute(0,2,1)   #CNN

        new_X_fre=self.reverse_cnn_fre(X_fre.permute(0,2,1)).permute(0,2,1)
#         X_fre,_=self.rnn(X_fre)
        length_fre=(torch.ones(X_fre.shape[0])*X_fre.shape[1]).to(X_fre.device).long()
        #make attention mask
        attention_mask=self.make_mask(X_fre,length_fre)
#         R=self.rpr_embedding(self.R_idx[:X_fre.shape[1],:X_fre.shape[1]])
        embedding_output=self.P_fre(X_fre)

        #adjust attention mask
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        #make head mask
        head_mask = [None] * self.config_fre.num_hidden_layers
        outputs_fre=self.Encoder_fre(  embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask)
        X_fre=outputs_fre[0]
#         fre_output=self.rnn_att_fre(X_fre)
#         X_fre=self.pooler(X_fre.permute(0,2,1)).permute(0,2,1)
#         attention_mask=self.make_mask(X_fre,length//4)
#         begin_hiddens,end_hiddens=[],[]
#         for i in range(1):
#             total_length=X_fre.shape[1]
#             new_X_fre=pack_padded_sequence(X_fre,length//4,batch_first=True,enforce_sorted=False)
#             hidden,_=self.rnns[i](new_X_fre)
#             hidden,_=pad_packed_sequence(hidden,batch_first=True)
#             hidden_avg = torch.sum(hidden * attention_mask.unsqueeze(1).transpose(2, 1), dim=1)\
#                                          / torch.sum(attention_mask, dim=1, keepdim=True)

#             hidden_max, _ = self.replace_masked(hidden, attention_mask, -1e7).max(dim=1)
#             begin_hiddens.append(hidden_avg)
#             end_hiddens.append(hidden_max)

#         begin_hidden=torch.cat(begin_hiddens,dim=-1)
#         end_hidden=torch.cat(end_hiddens,dim=-1)
#         hidden=torch.cat([begin_hidden,end_hidden],dim=-1)
        hidden=torch.cat([F.adaptive_avg_pool1d(X_fre.permute(0,2,1),1).squeeze(-1),\
                          F.adaptive_avg_pool1d(X_time.permute(0,2,1),1).squeeze(-1),\
                          F.adaptive_max_pool1d(X_fre.permute(0,2,1),1).squeeze(-1),\
                          F.adaptive_max_pool1d(X_time.permute(0,2,1),1).squeeze(-1),\
                            time_output],dim=-1)
        return hidden
class Seq2SeqEncoder(nn.Module):
    """
    RNN taking variable length padded sequences of vectors as input and
    encoding them into padded sequences of vectors of the same length.
    This module is useful to handle batches of padded sequences of vectors
    that have different lengths and that need to be passed through a RNN.
    The sequences are sorted in descending order of their lengths, packed,
    passed through the RNN, and the resulting sequences are then padded and
    permuted back to the original order of the input sequences.
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):
        """
        Args:
            rnn_type: The type of RNN to use as encoder in the module.
                Must be a class inheriting from torch.nn.RNNBase
                (such as torch.nn.LSTM for example).
            input_size: The number of expected features in the input of the
                module.
            hidden_size: The number of features in the hidden state of the RNN
                used as encoder by the module.
            num_layers: The number of recurrent layers in the encoder of the
                module. Defaults to 1.
            bias: If False, the encoder does not use bias weights b_ih and
                b_hh. Defaults to True.
            dropout: If non-zero, introduces a dropout layer on the outputs
                of each layer of the encoder except the last one, with dropout
                probability equal to 'dropout'. Defaults to 0.0.
            bidirectional: If True, the encoder of the module is bidirectional.
                Defaults to False.
        """
        assert issubclass(rnn_type, nn.RNNBase),            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        """
        Args:
            sequences_batch: A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
            sequences_lengths: A 1D tensor containing the sizes of the
                sequences in the input batch.
        Returns:
            reordered_outputs: The outputs (hidden states) of the encoder for
                the sequences in the input batch, in the same order.
        """
        total_length=sequences_batch.shape[1]
        packed_batch = nn.utils.rnn.pack_padded_sequence(sequences_batch,
                                                         sequences_lengths,
                                                         batch_first=True,enforce_sorted=False)

        outputs, _ = self._encoder(packed_batch, None)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True,total_length=total_length)

        return outputs
def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
class Residual_Attention(nn.Module):
    def __init__(self,key_size,dropout=0):
        super(Residual_Attention,self).__init__()
        self.W_k=nn.Sequential(nn.Linear(key_size*2,1,bias=False)
                              ,nn.Tanh())
        
#         self.v=nn.Linear(units,1,bias=False)
        self.dropout=nn.Dropout(dropout)
    def forward(self,sequence,valid_len=None):
        if valid_len is not None:
            valid_len -=1
            valid_len[valid_len==0] =1
        dis_seq = sequence[:,1:] - sequence[:,:-1]
        sim_seq = sequence[:,1:] * sequence[:,:-1]
        final_seq = torch.cat([dis_seq,sim_seq],dim=-1)
        key = final_seq
        if key.shape[1]==0:
            return torch.zeros(sequence.shape[0],sequence.shape[-1]*2).to(sequence.device)
        value = final_seq
        key=self.W_k(key)
        scores=key.squeeze(-1)
        attention_weights=self.dropout(masked_softmax(scores,valid_len)).unsqueeze(-1)
#         print(attention_weights)
        return (attention_weights*value).sum(dim=1)    
class Mutual_Attention(nn.Module):
    def __init__(self,embedding_dim,hidden_size,dropout=0):
        super(Mutual_Attention, self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)


        self._attention =DotProductAttention(self.dropout)

        self._projection = nn.Sequential(nn.Linear(4*self.embedding_dim,
                                                   self.hidden_size),
                                         nn.ReLU())
        #也许可以分成两个


        self._composition_h = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size+self.embedding_dim,
                                           self.hidden_size,
                                           bidirectional=True) 
        self._composition_p = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size+self.embedding_dim,
                                           self.hidden_size,
                                           bidirectional=True) 
        # 也许可以分成两个

    def make_mask(self,X,valid_len):
            shape=X.shape
            if valid_len.dim()==1:
                valid_len=valid_len.view(-1,1).repeat(1,shape[1])
            mask=(torch.arange(0,X.shape[1]).repeat(X.shape[0],1).to(X.device)<valid_len).float()
            return mask
    def replace_masked(self,tensor, mask, value):
        mask = mask.unsqueeze(1).transpose(2, 1)
        reverse_mask = 1.0 - mask
        values_to_add = value * reverse_mask
        return tensor * mask + values_to_add
    def forward(self,premises,hypotheses,premises_lengths, hypotheses_lengths):

        premises_mask = self.make_mask(premises, premises_lengths).to(premises.device)
        hypotheses_mask = self.make_mask(hypotheses, hypotheses_lengths).to(premises.device)

        embedded_premises = premises
        embedded_hypotheses = hypotheses

        encoded_premises = embedded_premises
        encoded_hypotheses = embedded_hypotheses
#         print(encoded_premises.shape)
#         print(encoded_hypotheses.shape)
        attended_premises, attended_hypotheses = self._attention(encoded_premises,encoded_hypotheses,
                                                 encoded_hypotheses, hypotheses_lengths),\
                                                 self._attention(encoded_hypotheses,encoded_premises,
                                                 encoded_premises,premises_lengths)
        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses - attended_hypotheses,
                                         encoded_hypotheses * attended_hypotheses],
                                        dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition_p(torch.cat([projected_premises,premises],dim=-1), premises_lengths)
        v_bj = self._composition_h(torch.cat([projected_hypotheses,hypotheses],dim=-1), hypotheses_lengths)


        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1).transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1).transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = self.replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = self.replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        hidden = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        return hidden

class EmbeddingLayer(nn.Module):
    def __init__(self,input_size,use_embedding = False,embedding_weight = None,num_embeddings = 10000,\
                use_time_position = False,pre_cnn_time_position = False,use_idx_embedding = False,idx_embedding_weight = None,\
                use_geo_position =False,pool = True, sinusoidal=False,use_layernorm =True,**kwargs):
        super(EmbeddingLayer,self).__init__()
        self.use_embedding = use_embedding
        self.use_geo_position = use_geo_position
        self.use_idx_embedding = use_idx_embedding
        if self.use_embedding:
            if embedding_weight is not None:
                self.embedding = nn.Embedding(embedding_weight.shape[0],embedding_weight.shape[1],padding_idx = 0,\
                                             _weight=embedding_weight)
                assert embedding_weight.shape[1] == input_size
            else:
                self.embedding = nn.Embedding(num_embeddings,input_size,padding_idx = 0)
        

        if self.use_idx_embedding:
            if idx_embedding_weight is not None:
                self.idx_embedding = nn.ModuleList([nn.Embedding(idx_embedding_weight[i].shape[0],idx_embedding_weight[i].shape[1],\
                                             _weight=idx_embedding_weight[i]) for i in range(3)])
            else:
                self.idx_embedding = nn.ModuleList([nn.Emebedding(500,input_size) for i in range(3)])
        
        if self.use_geo_position:
            config = BertConfig(hidden_size=input_size,max_position_embeddings=1000)
            self.geo_embedding = XY_Encoding(config, sinusoidal=sinusoidal,use_layernorm = use_layernorm)
    def forward(self,X,idx_list,geo_position_ids):
            X = self.embedding(X)
            X_embed = X
            if self.use_idx_embedding and idx_list is not None:
                X +=self.idx_embedding[0](idx_list[0]).unsqueeze(1)
                X +=self.idx_embedding[1](idx_list[1]).unsqueeze(1)
                X +=self.idx_embedding[2](idx_list[2]).unsqueeze(1)
                X_embed = X

            if self.use_geo_position:
                X = self.geo_embedding(X,geo_position_ids)
                X_embed = X
            return X_embed
class MLPModel(nn.Module):
    def __init__(self,input_size,input_size_fre,hidden_size,dropout_rate,bidirectional = False,\
                use_phase = False,use_rnn = True,num_hidden_layers=1,num_attention_heads=4,use_fre=True,
                use_cnn = True,use_embedding = False,embedding_weight = None,num_embeddings = 10000,num_cnn_layers = 3,\
                use_time_position = False,pre_cnn_time_position = False,use_idx_embedding = False,idx_embedding_weight = None,\
                use_geo_position =False,pool = True, sinusoidal=False,**kwargs):
        super(MLPModel,self).__init__()
        self.embedding = EmbeddingLayer(input_size,use_embedding,embedding_weight,num_embeddings,\
                use_time_position,pre_cnn_time_position,use_idx_embedding,idx_embedding_weight,\
                use_geo_position,pool, sinusoidal,use_layernorm=True)
        self.dropout=dropout_rate
        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            self.dense_dropout = nn.Dropout(self.dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Sequential(nn.Linear(input_size,input_size//2),nn.ReLU(),nn.Linear(input_size//2,2))
    def forward(self,X,X_fre,valid_len,time_dis,labels,time_position_ids = None,idx_list = None,geo_position_ids = None):
        X_embed = self.embedding(X,idx_list,geo_position_ids)
        inputs = self.pool(X_embed.permute(0,2,1)).squeeze(-1)
        logits = self.dense(inputs)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits,labels)      
        return logits,loss,inputs,X_embed,X_embed 
class CNNModel(nn.Module):
    def __init__(self,input_size,input_size_fre,hidden_size,dropout_rate,bidirectional = False,\
                use_phase = False,use_rnn = True,num_hidden_layers=1,num_attention_heads=4,use_fre=True,
                use_cnn = True,use_embedding = False,embedding_weight = None,num_embeddings = 10000,num_cnn_layers = 3,\
                use_time_position = False,pre_cnn_time_position = False,use_idx_embedding = False,idx_embedding_weight = None,\
                use_geo_position =False,pool = True, sinusoidal=False,**kwargs):
        super(CNNModel,self).__init__()
        self.embedding = EmbeddingLayer(input_size,use_embedding,embedding_weight,num_embeddings,\
                use_time_position,pre_cnn_time_position,use_idx_embedding,idx_embedding_weight,\
                use_geo_position,pool, sinusoidal,use_layernorm=True)
        self.dropout=dropout_rate
        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            self.dense_dropout = nn.Dropout(self.dropout)
        self.cnn = nn.Sequential()
        for i in range(num_cnn_layers):
                self.cnn.add_module(str(i),nn.Sequential(nn.Conv1d(input_size,hidden_size,3,padding=1,stride = 2),
                              nn.BatchNorm1d(hidden_size))) 
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Sequential(nn.Linear(hidden_size,hidden_size//2),nn.ReLU(),nn.Linear(hidden_size//2,2))
    def forward(self,X,X_fre,valid_len,time_dis,labels,time_position_ids = None,idx_list = None,geo_position_ids = None):
        X_embed = self.embedding(X,idx_list,geo_position_ids)
        inputs = self.pool(self.cnn(X_embed.permute(0,2,1))).squeeze(-1)
        logits = self.dense(inputs)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits,labels)      
        return logits,loss,inputs,X_embed,X_embed 
class BiGRUModel(nn.Module):
    def __init__(self,input_size,input_size_fre,hidden_size,dropout_rate,bidirectional = False,\
                use_phase = False,use_rnn = True,num_hidden_layers=1,num_attention_heads=4,use_fre=True,
                use_cnn = True,use_embedding = False,embedding_weight = None,num_embeddings = 10000,num_cnn_layers = 3,\
                use_time_position = False,pre_cnn_time_position = False,use_idx_embedding = False,idx_embedding_weight = None,\
                use_geo_position =False,pool = True, sinusoidal=False,**kwargs):
        super(BiGRUModel,self).__init__()
        self.embedding = EmbeddingLayer(input_size,use_embedding,embedding_weight,num_embeddings,\
                use_time_position,pre_cnn_time_position,use_idx_embedding,idx_embedding_weight,\
                use_geo_position,pool, sinusoidal,use_layernorm=True)
        self.dropout=dropout_rate
        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            self.dense_dropout = nn.Dropout(self.dropout)
        self.rnn = nn.GRU(input_size = input_size,hidden_size = hidden_size,\
                              num_layers = num_hidden_layers,batch_first = True,bidirectional = bidirectional)
        self.pool = nn.AdaptiveAvgPool1d(1)
        if bidirectional:
            hidden_size *=2
        self.dense = nn.Sequential(nn.Linear(hidden_size,hidden_size//2),nn.ReLU(),nn.Linear(hidden_size//2,2))
    def forward(self,X,X_fre,valid_len,time_dis,labels,time_position_ids = None,idx_list = None,geo_position_ids = None):
        X_embed = self.embedding(X,idx_list,geo_position_ids)
        inputs = self.pool(self.rnn(X_embed)[0].permute(0,2,1)).squeeze(-1)
        logits = self.dense(inputs)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits,labels)      
        return logits,loss,inputs,X_embed,X_embed 
class ConvGRU_Encoder(nn.Module):
    def __init__(self,input_size,input_size_fre,hidden_size,dropout_rate,bidirectional = False,\
                use_phase = False,use_rnn = True,num_hidden_layers=1,num_attention_heads=4,use_fre=True,
                use_cnn = True,use_embedding = False,embedding_weight = None,num_embeddings = 10000,num_cnn_layers = 3,\
                use_time_position = False,pre_cnn_time_position = False,use_idx_embedding = False,idx_embedding_weight = None,\
                use_geo_position =False,pool = True, sinusoidal=False,**kwargs):
        super(ConvGRU_Encoder,self).__init__()
        self.dropout=dropout_rate
        LayerNorm=nn.LayerNorm
        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            self.dense_dropout = nn.Dropout(self.dropout)
            
        self.use_embedding = use_embedding
        self.use_geo_position = use_geo_position
        self.use_idx_embedding = use_idx_embedding
        
        self.use_time_position = use_time_position
        self.pre_cnn_time_position = pre_cnn_time_position

        self.use_cnn = use_cnn
        self.use_rnn = use_rnn
        self.use_fre = use_fre
        
        if self.use_embedding:
            if embedding_weight is not None:
                self.embedding = nn.Embedding(embedding_weight.shape[0],embedding_weight.shape[1],padding_idx = 0,\
                                             _weight=embedding_weight)
                assert embedding_weight.shape[1] == input_size
            else:
                self.embedding = nn.Embedding(num_embeddings,input_size,padding_idx = 0)
        

        if self.use_idx_embedding:
            if idx_embedding_weight is not None:
                self.idx_embedding = nn.ModuleList([nn.Embedding(idx_embedding_weight[i].shape[0],idx_embedding_weight[i].shape[1],\
                                             _weight=idx_embedding_weight[i]) for i in range(3)])
            else:
                self.idx_embedding = nn.ModuleList([nn.Emebedding(500,input_size) for i in range(3)])
        
        if self.use_geo_position:
            config = BertConfig(hidden_size=input_size,max_position_embeddings=1000)
            self.geo_embedding = XY_Encoding(config, sinusoidal=sinusoidal)
                

        self.num_cnn_layers = num_cnn_layers 
        self.pool = pool
        if self.use_cnn:
            self.cnn = nn.Sequential()
            for i in range(num_cnn_layers):
                if pool:
                    self.cnn.add_module(str(i),nn.Sequential(nn.Conv1d(input_size,hidden_size,3,padding=1),
                                  nn.BatchNorm1d(hidden_size),
                                  nn.ReLU(),
                                  nn.MaxPool1d(2,padding=0,return_indices=True)))
                else:
                    self.cnn.add_module(str(i),nn.Sequential(nn.Conv1d(input_size,hidden_size,3,padding=1,stride = 2),
                                  nn.BatchNorm1d(hidden_size)))                    

        if self.use_rnn:
            if self.use_cnn:
                tmp_input_size = hidden_size
            else:
                tmp_input_size = input_size
            self.rnn = nn.GRU(input_size = tmp_input_size,hidden_size = hidden_size,\
                              num_layers = num_hidden_layers,batch_first = True,bidirectional = bidirectional)
        else:
            self.rnn = Transformer_Encoder(tmp_input_size,num_hidden_layers,num_attention_heads)
            
            

        if use_phase:
            input_size_fre *= 2

        if self.use_fre:
            self.cnn_fre = nn.Sequential(nn.Conv1d(input_size_fre,hidden_size,3,padding=1),
                          nn.BatchNorm1d(hidden_size),
                          nn.ReLU(),
                          nn.MaxPool1d(2,padding=0),
                          nn.Conv1d(hidden_size,hidden_size,3,padding=1),
                          nn.BatchNorm1d(hidden_size),
                          nn.ReLU(),
                          nn.MaxPool1d(2,padding=1))
#         self.rnn_fre=nn.GRU(input_size=input_size_fre,hidden_size=hidden_size,batch_first=True,bidirectional=bidirectional)

#         self.attunit=MLPAttention_weight(hidden_size,16,16*4,0)

        self.hidden_size_time = hidden_size*(2 if self.use_rnn and self.rnn.bidirectional else 1)
        self.hidden_size_fre = hidden_size*(2 if 'rnn_fre' in self.__dict__['_modules'] and self.rnn_fre.bidirectional else 1)
        
        output_size_time = self.hidden_size_time*(2 if self.use_rnn and self.rnn.bidirectional else 1)
        output_size_fre = self.hidden_size_fre*(2 if 'rnn_fre' in self.__dict__['_modules'] and self.rnn_fre.bidirectional else 1)


        self.ln_rnn = LayerNorm(self.hidden_size_time)
        self.ln_fre = LayerNorm(self.hidden_size_fre)
    def get_embedding(self,X,idx_list,geo_position_ids):
            X = self.embedding(X)
            X_embed = X
            if self.use_idx_embedding and idx_list is not None:
                X +=self.idx_embedding[0](idx_list[0]).unsqueeze(1)
                X +=self.idx_embedding[1](idx_list[1]).unsqueeze(1)
                X +=self.idx_embedding[2](idx_list[2]).unsqueeze(1)
                X_embed = X

            if self.use_geo_position:
                X = self.geo_embedding(X,geo_position_ids)
                X_embed = X
            return X_embed
    def forward(self,X,X_fre,valid_len,time_dis,labels,time_position_ids = None,idx_list = None,geo_position_ids = None,\
                initial_state = None,input_embed = None):
        if self.use_embedding and input_embed is None:
            X_embed = self.get_embedding(X,idx_list,geo_position_ids)
            X = X_embed
        elif input_embed is not None:
            X = input_embed
            X_embed = X
        indices_list = []
        if self.use_cnn:
            X = X.permute(0,2,1)
            if not self.pool:
                X = self.cnn(X).permute(0,2,1)   #CNN
            else:
                for e in self.cnn:
                    X,indice = e(X)
                    indices_list.append(indice)
                X = X.permute(0,2,1)

        if self.dropout and self.use_rnn:
            X = self._rnn_dropout(X)
#             X_fre=self._rnn_dropout(X_fre)
        if self.use_fre:
            X_fre = self.dense_dropout(X_fre)

    #         Y_fre,_=self.rnn_fre(X_fre) #RNN
    #         Y_fre=self.ln_fre(Y_fre)

            Y_fre = F.adaptive_avg_pool1d(self.cnn_fre(X_fre.permute(0,2,1)),1).permute(0,2,1) #CNN
            Y_fre_hidden = Y_fre[:,-1]
        if self.use_cnn:
            valid_len = valid_len//pow(2,self.num_cnn_layers)
            valid_len[valid_len==0] = 1

        if self.use_rnn:
            inputs = nn.utils.rnn.pack_padded_sequence(X,valid_len,batch_first=True,enforce_sorted=False)
            if initial_state is not None:
                if self.rnn.num_layers >1:
                    tmp = torch.zeros(self.rnn.num_layers-1,initial_state.shape[0],initial_state.shape[1]).to(X.device)
                    initial_state = torch.cat([initial_state.unsqueeze(0),tmp],dim=0)
                else:
                    initial_state = initial_state.unsqueeze(0)
#             self.rnn.flatten_parameters()
            Y_time,_ = self.rnn(inputs,initial_state)
            Y_time,d = nn.utils.rnn.pad_packed_sequence(Y_time,batch_first=True,total_length = X.shape[1])
            

            _ = _.view(self.rnn.num_layers,1+(int)(self.rnn.bidirectional),Y_time.shape[0],self.rnn.hidden_size)
            
            Y_time = self.ln_rnn(Y_time)
            Y_time_end = torch.gather(Y_time,dim=1,index=(valid_len-1).unsqueeze(1).repeat(1,self.hidden_size_time).unsqueeze(1))[:,0]

            if self.rnn.bidirectional:
                Y_time_hidden = torch.cat([Y_time[:,0],Y_time_end],dim=-1)
            else:
                Y_time_hidden = Y_time_end
            
            sequence_logits = Y_time
            final_state = _
            
        else:
            # decode 的mask已经写好了，缺的是把query的inital state传进去，这一部分忘记怎么处理了
            position_encode = 1 - self.pre_cnn_time_position
            if initial_state is not None:
                decode = False
            else:
                decode = True
            if self.use_time_position:
                Y_time_hidden,sequence_logits = self.rnn(X,valid_len,position_encode,time_position_ids,\
                                                        decode = decode,query_state = initial_state)
            else:
                Y_time_hidden,sequence_logits  = self.rnn(X,valid_len,position_encode,\
                                                         decode = decode,query_state = initial_state)
            final_state = None
        
#         Y_time_hidden=self.self_att(Y_time[:,-1].unsqueeze(dim=1),Y_time,Y_time).squeeze(dim=1)
#         Y_time_hidden=Y_time[:,-1]
        if self.use_fre:
            Y_hidden = torch.cat([Y_time_hidden,Y_fre_hidden],dim=-1)  #前后cat
        else:
            Y_hidden = Y_time_hidden
        return Y_hidden,sequence_logits,final_state,X_embed,indices_list
class ConvGRU(nn.Module):
    def __init__(self,input_size,input_size_fre,hidden_size,dropout_rate,bidirectional = False,\
                use_phase = False,use_rnn = True,num_hidden_layers=1,num_attention_heads=4,use_fre=True,
                use_cnn = True,use_embedding = False,embedding_weight = None,num_embeddings = 10000,num_cnn_layers = 3,\
                use_time_position = False,pre_cnn_time_position = False,use_idx_embedding = False,idx_embedding_weight = None,\
                use_geo_position =False,sinusoidal=False,pretrain = False,use_embed = False,**kwargs):
        super(ConvGRU,self).__init__()
        self.dropout=dropout_rate
        LayerNorm=nn.LayerNorm
        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            self.dense_dropout = nn.Dropout(self.dropout) 
        self.use_fre = use_fre
        
        self.encoder = ConvGRU_Encoder(input_size,input_size_fre,hidden_size,dropout_rate,bidirectional,\
                use_phase,use_rnn ,num_hidden_layers,num_attention_heads,use_fre,
                use_cnn ,use_embedding ,embedding_weight ,num_embeddings ,num_cnn_layers ,\
                use_time_position ,pre_cnn_time_position ,use_idx_embedding,idx_embedding_weight ,\
                use_geo_position,sinusoidal=sinusoidal,**kwargs)
        
        self.hidden_size_time = hidden_size*(2 if use_rnn and bidirectional else 1)
        self.hidden_size_fre = hidden_size*(2 if 'rnn_fre' in self.__dict__['_modules'] and bidirectional else 1)
        
        output_size_time = self.hidden_size_time*(2 if use_rnn and bidirectional else 1)
        output_size_fre = self.hidden_size_fre*(2 if 'rnn_fre' in self.__dict__['_modules'] and bidirectional else 1)
        
        if self.use_fre:
            self.dense = nn.Sequential(nn.Linear(output_size_time+output_size_fre,hidden_size),
                                         self.dense_dropout,nn.ReLU(),nn.Linear(hidden_size,2))
        else:
            self.dense = nn.Sequential(nn.Linear(output_size_time,hidden_size),
                                         self.dense_dropout,nn.ReLU(),nn.Linear(hidden_size,2))
        self.use_cnn = use_cnn
        self.num_cnn_layers = num_cnn_layers
        if self.use_cnn:
            self.cnn_trans = nn.Sequential()
            for i in range(num_cnn_layers):
                if i ==0:
                    i_size = input_size
                    o_size = self.hidden_size_time
                else:
                    i_size = hidden_size
                    o_size = input_size
                if self.encoder.pool:
                    self.cnn_trans.add_module(str(i)+"_0",nn.MaxUnpool1d(2,padding=0),)
                    self.cnn_trans.add_module(str(i)+"_1",\
                                              nn.ConvTranspose1d(in_channels=o_size,out_channels=input_size,kernel_size=3,stride=1,padding=1))
                else:
                    self.cnn_trans.add_module(str(i),\
                                              nn.Sequential(nn.ConvTranspose1d(in_channels=o_size,out_channels=input_size,kernel_size=3,stride=2,padding=1)))
        self.pretrain = pretrain
        self.use_embed = use_embed
        self.trans = nn.Linear(input_size,input_size,bias = False)

    def forward(self,X,X_fre,valid_len,time_dis,labels,time_position_ids = None,idx_list = None,geo_position_ids = None):
        inputs,sequence_logits,final_state,X_embed,indices_list = self.encoder(X,X_fre,valid_len,time_dis,\
                                 labels,time_position_ids,idx_list,geo_position_ids)
        if self.use_cnn and self.pretrain and not self.use_embed:
            x = sequence_logits.permute(0,2,1)
            for i,e in enumerate(self.cnn_trans):
                if i%2==0:
                    idx = indices_list[-(i//2+1)]
                    if i==0:
                        idx = torch.repeat_interleave(idx,2,dim=1)
                    x = e(x,idx)
                else:
                    x = e(x)
            x = x.permute(0,2,1)
            assert x.shape == X_embed.shape
            
        elif self.use_embed:
            x = X_embed
        else:
            x = sequence_logits
        
        logits = self.dense(inputs)
        loss = torch.zeros([]).to(X.device)
        if self.pretrain:
            for x_y,embed,length in zip(geo_position_ids,x,valid_len):
                if length<=3:
                    continue
                new_x_y,counts = torch.unique_consecutive(x_y[:length],return_counts=True,dim=0)
                idx = torch.cat([torch.tensor([0]).to(x_y.device),torch.cumsum(counts,0)[:-1]])
                embed = embed[idx]
                if embed.shape[0]<3:
                    continue
                new_x_y = new_x_y[1:] - new_x_y[:-1]
                embed = embed[1:] - embed[:-1]
                embed = self.trans(embed)
                label_angle = torch.sum(new_x_y[1:]*new_x_y[:-1],dim=-1).float()/(torch.sqrt((new_x_y[1:]**2).sum(dim=-1).float())*torch.sqrt((new_x_y[:-1]**2).sum(dim=-1).float())+1e-6)
                predict_angle = torch.sum(embed[1:]*embed[:-1],dim=-1).float()/(torch.sqrt((embed[1:]**2).sum(dim=-1).float())*torch.sqrt((embed[:-1]**2).sum(dim=-1).float())+1e-6)
                l = F.smooth_l1_loss(predict_angle,label_angle)
                loss +=l
        else:

            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits,labels)

        return logits,loss,inputs,sequence_logits,X_embed
class FusionModel(nn.Module):
    def __init__(self,location_config,mouse_config,location_hidden_size,mouse_hidden_size,\
                 use_mutual_attention = False,use_residual = False,use_rnn_output = False,\
                 pretrain = False,use_embed = False,handle_mis = False,model_type = "ConvGRU",tri_loss = True):
        super(FusionModel,self).__init__()
        
        self.tri_loss = tri_loss
        self.handle_mis = handle_mis
        self.ptretrain = pretrain
        self.model_type = model_type
        if self.model_type == "ConvGRU":
            self.location_encoder = ConvGRU(**location_config,pretrain = pretrain,use_embed = use_embed)
            self.mouse_encoder = ConvGRU(**mouse_config,pretrain = pretrain,use_embed = use_embed)
        elif self.model_type == 'MLP':
            self.location_encoder = MLPModel(**location_config)
            self.mouse_encoder = MLPModel(**mouse_config)
        elif self.model_type == 'CNN':
            self.location_encoder = CNNModel(**location_config)
            self.mouse_encoder = CNNModel(**mouse_config)
        elif self.model_type == 'BiGRU':
            self.location_encoder = BiGRUModel(**location_config)
            self.mouse_encoder = BiGRUModel(**mouse_config)            
        
        self.use_mutual_attention = use_mutual_attention
        if self.use_mutual_attention:
            embed_dim = location_config['hidden_size']*(2 if location_config['use_rnn'] and location_config['bidirectional'] else 1)
            hidden_size = location_hidden_size//4
            self.att = Mutual_Attention(embed_dim,hidden_size,location_config['dropout_rate'])
        self.use_residual = use_residual
        self.use_rnn_output = use_rnn_output
        if self.use_residual:
            rnn_size = location_config['hidden_size']*(2 if location_config['use_rnn'] and location_config['bidirectional'] else 1)
            embed_dim = rnn_size if self.use_rnn_output else location_config['input_size']
            self.embed_dim = embed_dim
            self.res_att_l = Residual_Attention(embed_dim,location_config['dropout_rate'])
            self.res_att_m = Residual_Attention(embed_dim,mouse_config['dropout_rate'])
        hidden_size = location_hidden_size + mouse_hidden_size
        if self.use_residual:
            hidden_size += self.embed_dim * 4
        self.dense = nn.Sequential(nn.Linear(hidden_size,hidden_size//2),nn.ReLU(),nn.Linear(hidden_size//2,hidden_size//4),nn.ReLU(),\
                                  nn.Linear(hidden_size//4,2))

    def forward(self,X_loc,X_fre_loc,valid_len_loc,time_dis_loc,\
                X_mo,X_fre_mo,valid_len_mo,time_dis_mo,labels,\
                time_position_ids_loc = None,idx_list_loc = None,geo_position_ids_loc = None,\
                time_position_ids_mo = None,idx_list_mo = None,geo_position_ids_mo = None,*args,**kwargs):
        location_logits,location_loss,location_hidden,location_sequence_logits,location_embed  = self.location_encoder(X_loc,X_fre_loc,valid_len_loc,time_dis_loc,\
                                                                               labels,time_position_ids_loc,idx_list_loc,geo_position_ids_loc)
        mouse_logits,mouse_loss,mouse_hidden,mouse_sequence_logits,mouse_embed  = self.mouse_encoder(X_mo,X_fre_mo,valid_len_mo,time_dis_mo,\
                                                                               labels,time_position_ids_mo,idx_list_mo,geo_position_ids_mo)
        new_valid_len_loc = valid_len_loc.clone()
        new_valid_len_mo = valid_len_mo.clone()
        if self.ptretrain:
            logits = location_logits
            hidden = location_hidden
            loss = location_loss + mouse_loss
        else:
            if self.model_type == 'ConvGRU':
                if self.location_encoder.encoder.use_cnn:
                    new_valid_len_loc = (new_valid_len_loc)//pow(2,self.location_encoder.encoder.num_cnn_layers)
                    new_valid_len_loc[new_valid_len_loc==0] +=1

                if self.mouse_encoder.encoder.use_cnn:
                    new_valid_len_mo = (new_valid_len_loc)//pow(2,self.mouse_encoder.encoder.num_cnn_layers)
                    new_valid_len_mo[new_valid_len_mo==0] +=1

                if self.use_mutual_attention:
                    hidden = self.att(location_sequence_logits,mouse_sequence_logits,new_valid_len_loc,new_valid_len_mo)
                else:
                    hidden = torch.cat([location_hidden,mouse_hidden],dim=-1)

                if self.use_residual:
                    if self.use_rnn_output:
                        rh_l = self.res_att_l(location_sequence_logits,new_valid_len_loc)
                        rh_m = self.res_att_m(mouse_sequence_logits,new_valid_len_mo)
                    else:
                        rh_l = self.res_att_l(location_embed,valid_len_loc)
                        rh_m = self.res_att_m(mouse_embed,valid_len_mo)
                    hidden = torch.cat([hidden,rh_l,rh_m],dim=-1)
            else:
                    hidden = torch.cat([location_hidden,mouse_hidden],dim=-1)
            loss_func = nn.CrossEntropyLoss()
            logits  = self.dense(hidden)
            if self.tri_loss:
                loss = location_loss + mouse_loss + loss_func(logits,labels)
            else:
                loss = loss_func(logits,labels)
        if self.handle_mis:
            logits[valid_len_loc==1] = mouse_logits[valid_len_loc==1]
            logits[valid_len_mo==1] = location_logits[valid_len_mo==1]
        return logits,location_logits,mouse_logits,loss,hidden
class LatentGaussianMixture(nn.Module):
    def __init__(self, token_dim,rnn_dim,cluster_num,pretrain_dir = None,model = None):
        super(LatentGaussianMixture,self).__init__()
#         self.args = args
        self.token_dim = token_dim
        self.cluster_num = cluster_num
        self.rnn_dim = rnn_dim
        if pretrain_dir:
            mu_c_path = '{}/{}_{}_{}_{}/init_mu_c.npz'.format(pretrain_dir, model, 
                    token_dim, rnn_dim, cluster_num)
            mu_c = np.load(mu_c_path)
            self.mu_c = nn.Parameter(torch.tensor(mu_c,dtype = torch.float)) 
#             tf.get_variable("mu_c", initializer=tf.constant(mu_c))
        else:
            self.mu_c = nn.Parameter(torch.rand(cluster_num,rnn_dim,dtype = torch.float)) 
#             tf.get_variable("mu_c", [args.cluster_num, args.rnn_dim],
#                     initializer=tf.random_uniform_initializer(0.0, 1.0))

        self.log_sigma_sq_c = nn.Parameter(torch.zeros(cluster_num,rnn_dim,dtype = torch.float),requires_grad = False) 
#     tf.get_variable("sigma_sq_c", [args.cluster_num, args.rnn_dim],
#                 initializer=tf.constant_initializer(0.0), trainable=False)

        self.fc_mu_z = nn.Linear(token_dim,rnn_dim)
#     fc(args.rnn_dim, activation=None, use_bias=True,
#                 kernel_initializer=w_init, bias_initializer=b_init)

        self.fc_sigma_z = nn.Linear(token_dim,rnn_dim)
#     fc(args.rnn_dim, activation=None, use_bias=True,
#                 kernel_initializer=w_init, bias_initializer=b_init)

    def forward(self, embeded_state, return_loss=False):
#         args = self.args

        mu_z = self.fc_mu_z(embeded_state)
        log_sigma_sq_z = self.fc_sigma_z(embeded_state)

        eps_z = torch.normal(mean = 0.0,std = torch.ones(log_sigma_sq_z.shape)).to(embeded_state.device)
#         tf.random_normal(shape=tf.shape(log_sigma_sq_z), mean=0.0, stddev=1.0, dtype=tf.float32)
        z = mu_z + torch.sqrt(torch.exp(log_sigma_sq_z)) * eps_z 
#     mu_z + tf.sqrt(tf.exp(log_sigma_sq_z)) * eps_z

        stack_z = z.unsqueeze(1).repeat(1,self.cluster_num,1)
#     tf.stack([z] * args.cluster_num, axis=1)
        stack_mu_c = self.mu_c.unsqueeze(0).repeat(embeded_state.shape[0],1,1)
#     tf.stack([self.mu_c] * args.batch_size, axis=0)
        stack_mu_z = mu_z.unsqueeze(1).repeat(1,self.cluster_num,1)
#     tf.stack([mu_z] * args.cluster_num, axis=1)
        stack_log_sigma_sq_c = self.log_sigma_sq_c.unsqueeze(0).repeat(embeded_state.shape[0],1,1)
#     tf.stack([self.log_sigma_sq_c] * args.batch_size, axis=0)
        stack_log_sigma_sq_z = log_sigma_sq_z.unsqueeze(1).repeat(1,self.cluster_num,1)
#     tf.stack([log_sigma_sq_z] * args.cluster_num, axis=1)

        pi_post_logits = - torch.sum((stack_z - stack_mu_c)**2 / torch.exp(stack_log_sigma_sq_c), dim=-1)
#     - tf.reduce_sum(tf.square(stack_z - stack_mu_c) / tf.exp(stack_log_sigma_sq_c), axis=-1)

        pi_post = torch.softmax(pi_post_logits,dim=-1) + 1e-10
#     tf.nn.softmax(pi_post_logits) + 1e-10

        if not return_loss:
            return z
        else:
            batch_gaussian_loss = 0.5 * torch.sum(pi_post*torch.mean(stack_log_sigma_sq_c\
                                        +torch.exp(stack_log_sigma_sq_z)/torch.exp(stack_log_sigma_sq_c)\
                                        +(stack_mu_z - stack_mu_c)**2/torch.exp(stack_log_sigma_sq_c),dim=-1) ,dim = -1)\
                                        - 0.5 * torch.mean(1+log_sigma_sq_z,dim = -1)
# 0.5 * tf.reduce_sum(pi_post * tf.reduce_mean(stack_log_sigma_sq_c
#                         + tf.exp(stack_log_sigma_sq_z) / tf.exp(stack_log_sigma_sq_c)
#                         + tf.square(stack_mu_z - stack_mu_c) / tf.exp(stack_log_sigma_sq_c), axis=-1)
#                     , axis=-1) - 0.5 * tf.reduce_mean(1 + log_sigma_sq_z, axis=-1)

            batch_uniform_loss = torch.mean(torch.sum(pi_post*(torch.log(pi_post)+np.log(self.cluster_num)),dim=-1)) #我认为的形式
            #torch.mean(torch.mean(pi_post,dim=0)*(torch.log(torch.mean(pi_post,dim=0))+np.log(self.cluster_num)))
            # 
#     tf.reduce_mean(tf.reduce_mean(pi_post, axis=0) * tf.log(tf.reduce_mean(pi_post, axis=0)))  #这一步存疑
            return z, [batch_gaussian_loss, batch_uniform_loss]
class ConvGRU_VAE(nn.Module):
    def __init__(self,input_size,input_size_fre,hidden_size,dropout_rate,bidirectional = False,\
                use_phase = False,use_rnn = True,num_hidden_layers=1,num_attention_heads=4,use_fre=True,
                use_cnn = False,use_embedding = False,embedding_weight = None,num_embeddings = 10000,num_cnn_layers = 3,\
                use_time_position = False,pre_cnn_time_position = False,use_idx_embedding = False,idx_embedding_weight = None,\
                use_geo_position =False,cluster_num = 5,**kwargs):
        super(ConvGRU_VAE,self).__init__()

        self.loss_type = "mse"
        self.mode = "train"

        self.dropout=dropout_rate
        LayerNorm=nn.LayerNorm
        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            self.dense_dropout = nn.Dropout(self.dropout) 
        self.use_fre = use_fre
        
        self.encoder = ConvGRU_Encoder(input_size,input_size_fre,hidden_size,dropout_rate,bidirectional,\
                use_phase,use_rnn ,num_hidden_layers,num_attention_heads,use_fre,
                use_cnn ,use_embedding ,embedding_weight ,num_embeddings ,num_cnn_layers ,\
                use_time_position ,pre_cnn_time_position ,use_idx_embedding,idx_embedding_weight ,\
                use_geo_position,pool = False,**kwargs)

        self.decoder = ConvGRU_Encoder(input_size,input_size_fre,hidden_size,dropout_rate,False,\
                use_phase,use_rnn ,num_hidden_layers,num_attention_heads,use_fre,
                False ,use_embedding ,embedding_weight ,num_embeddings ,num_cnn_layers ,\
                use_time_position ,pre_cnn_time_position ,use_idx_embedding,idx_embedding_weight ,\
                use_geo_position,pool = False,**kwargs)
        
        self.hidden_size_time = hidden_size*(2 if use_rnn and bidirectional else 1)
        self.hidden_size_fre = hidden_size*(2 if 'rnn_fre' in self.__dict__['_modules'] and bidirectional else 1)
        
        output_size_time = self.hidden_size_time*(2 if use_rnn and bidirectional else 1)
        output_size_fre = self.hidden_size_fre*(2 if 'rnn_fre' in self.__dict__['_modules'] and bidirectional else 1)
        
        self.use_cnn = use_cnn
        self.cluster_num = cluster_num
        self.rnn_dim = hidden_size
        self.latent_space = LatentGaussianMixture(hidden_size,hidden_size,cluster_num)

        
        self.dense = nn.Sequential(nn.Linear(output_size_time,hidden_size),
                                     self.dense_dropout,nn.ReLU(),nn.Linear(hidden_size,2))

        self.loss_bn = nn.BatchNorm1d(1)

        self.logits_dense = nn.Linear(output_size_time,self.encoder.embedding.weight.shape[0])

    def make_mask(self,X,valid_len):
        shape=X.shape[:2]
        if valid_len.dim() == 1:
            valid_len = valid_len.view(-1,1).repeat(1,shape[1])
        mask=(torch.arange(0,X.shape[1]).repeat(X.shape[0],1).to(X.device)<valid_len).float()
        return mask
    def forward(self,X,X_fre,valid_len,time_dis,labels,time_position_ids = None,idx_list = None,geo_position_ids = None):
        masks = self.make_mask(X,valid_len)
        batch_zeros = torch.zeros(X.shape[0], 1).long().to(X.device)
        targets = torch.cat([X, batch_zeros], dim=1)
        tokens = torch.cat([batch_zeros, X], dim=1)
        masks = torch.cat([masks, batch_zeros.float()], dim=1)
        new_valid_len = valid_len + 1
        new_time_dis = torch.cat([batch_zeros,time_dis],dim=1)
        new_time_position_ids = torch.cat([batch_zeros, time_position_ids], dim=1)
        new_geo_position_ids = torch.cat([batch_zeros.unsqueeze(-1).repeat(1,1,2), geo_position_ids], dim=1)        
        if self.mode =='test':
            X_embed = self.encoder.get_embedding(X,idx_list,geo_position_ids)
            new_X_embed = torch.cat([torch.zeros(X.shape[0],1,X_embed.shape[-1]).float().to(X.device),X_embed],dim=1)

        
        if self.mode !="test":
            inputs,sequence_logits,final_state,X_embed,indices_list = self.encoder(X,X_fre,valid_len,time_dis,\
                                     labels,time_position_ids,idx_list,geo_position_ids)
            new_X_embed = torch.cat([torch.zeros(X.shape[0],1,X_embed.shape[-1]).float().to(X.device),X_embed],dim=1)
            outputs = sequence_logits
            encoder_final_state = final_state[0][0]

            z, latent_losses = self.latent_space(encoder_final_state, return_loss=True)
            outputs,output_sequence_logits,output_final_state,X_embed,indices_list = self.decoder(tokens,X_fre,new_valid_len,new_time_dis,labels,new_time_position_ids,idx_list,new_geo_position_ids,\
                                   initial_state=z,input_embed = new_X_embed)
            
            if self.mode == 'train' or self.mode == 'pretrain':
                res = self.loss(output_sequence_logits, targets, masks, latent_losses)
                res += [z]
            elif self.mode == 'eval':
                res = [self.anomaly_score(output_sequence_logits, targets, masks)]
                res +=[None,z]
            return res
        else:
            z_list = self.latent_space.mu_c.data
            res = [0,0,0]
            score_list = []
            for z in z_list:
                outputs,output_sequence_logits,output_final_state,X_embed = self.decoder(tokens,X_fre,new_valid_len,new_time_dis,labels,new_time_position_ids,idx_list,new_geo_position_ids,\
                                       initial_state=z.unsqueeze(0).repeat(tokens.shape[0],1),input_embed = new_X_embed)
                score = self.anomaly_score(output_sequence_logits, targets, masks)[:,0]
                score_list.append(score)
            scores = torch.max(torch.stack(score_list,dim=-1),dim=-1).values
            scores = torch.stack([scores,1-scores],dim=-1)
            res[0] = scores
            res[1] = None
            return res
    def anomaly_score(self, outputs, targets, masks):

        target_out_w = self.logits_dense.weight.data[targets]
        target_out_b = self.logits_dense.bias.data[targets]
        score =  torch.sum(masks*torch.sigmoid(torch.sum(outputs*target_out_w,dim=-1)+target_out_b),dim=-1)/torch.sum(masks,dim=-1)
        score = torch.stack([score,1-score],dim=-1)
#         tf.reduce_sum(
#                 masks * tf.exp(tf.log_sigmoid\(
#                     tf.reduce_sum(outputs * target_out_w, axis=-1) + target_out_b
#                     )), axis=-1, name="anomaly_score") / tf.reduce_sum(masks, axis=-1)
        return score

    def loss(self, outputs, targets, masks, latent_losses):
        batch_gaussian_loss, batch_uniform_loss = latent_losses
        prob = self.logits_dense(outputs)
        log_prob = F.log_softmax(prob,dim=-1)
        score = torch.sum(masks*torch.sigmoid(prob.gather(dim=-1,index =targets.unsqueeze(-1))).squeeze(-1),dim=-1)/torch.sum(masks,dim=-1)

        log_prob = log_prob.gather(dim=-1,index = targets.unsqueeze(-1)).squeeze(-1)
        log_prob[masks==0] = 0
        batch_rec_loss =   -log_prob.mean(dim=-1)/masks.mean(dim=-1)
#         tf.reduce_mean(
#                 tf.cast(masks, tf.float32) * tf.reshape(
#                     tf.nn.sampled_softmax_loss(
#                         weights=self.out_w,
#                         biases=self.out_b,
#                         labels=tf.reshape(targets, [-1, 1]),
#                         inputs=tf.reshape(outputs, [-1, args.rnn_dim]),
#                         num_sampled=args.num_negs,
#                         num_classes=self.out_size
#                     ), [args.batch_size, -1]
#                 ), axis=-1)

        rec_loss = torch.mean(batch_rec_loss)
        gaussian_loss = torch.mean(batch_gaussian_loss)
        uniform_loss = torch.mean(batch_uniform_loss)

        if self.cluster_num == 1:
            loss = rec_loss + gaussian_loss
        else:
#             print(rec_loss,gaussian_loss,uniform_loss)
            loss = 1.0 * rec_loss + 1.0 * gaussian_loss +  uniform_loss
            #这个地方也是存疑，明明在loss里面rnn_dim已经mean过了
            # loss = 1.0 * rec_loss + 1.0 / self.rnn_dim * gaussian_loss + 0.1 * uniform_loss

        pretrain_loss = rec_loss
        
        score = torch.stack([score,1-score],dim=-1)
#         tmp_score = self.anomaly_score( outputs, targets, masks)
        return [score,loss]
class ConvGRU_AutoEncoder(nn.Module):
    def __init__(self,input_size,input_size_fre,hidden_size,dropout_rate,bidirectional = False,\
                use_phase = False,use_rnn = True,num_hidden_layers=1,num_attention_heads=4,use_fre=True,
                use_cnn = True,use_embedding = False,embedding_weight = None,num_embeddings = 10000,num_cnn_layers = 3,\
                use_time_position = False,pre_cnn_time_position = False,use_idx_embedding = False,idx_embedding_weight = None,\
                use_geo_position =False,mse_loss = True,logloss = False,**kwargs):
        super(ConvGRU_AutoEncoder,self).__init__()

        if mse_loss:
            self.loss_type = "mse"
        else:
            self.loss_type = "logloss"

        self.dropout=dropout_rate
        LayerNorm=nn.LayerNorm
        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            self.dense_dropout = nn.Dropout(self.dropout) 
        self.use_fre = use_fre
        
        self.encoder = ConvGRU_Encoder(input_size,input_size_fre,hidden_size,dropout_rate,bidirectional,\
                use_phase,use_rnn ,num_hidden_layers,num_attention_heads,use_fre,
                use_cnn ,use_embedding ,embedding_weight ,num_embeddings ,num_cnn_layers ,\
                use_time_position ,pre_cnn_time_position ,use_idx_embedding,idx_embedding_weight ,\
                use_geo_position,pool = False,**kwargs)
        
        self.hidden_size_time = hidden_size*(2 if use_rnn and bidirectional else 1)
        self.hidden_size_fre = hidden_size*(2 if 'rnn_fre' in self.__dict__['_modules'] and bidirectional else 1)
        
        output_size_time = self.hidden_size_time*(2 if use_rnn and bidirectional else 1)
        output_size_fre = self.hidden_size_fre*(2 if 'rnn_fre' in self.__dict__['_modules'] and bidirectional else 1)
        
        self.use_cnn = use_cnn
        self.num_cnn_layers = 3
        if self.use_cnn:
            self.cnn_trans = nn.Sequential()
            for i in range(num_cnn_layers):
                if i ==0:
                    i_size = input_size
                    o_size = self.hidden_size_time
                else:
                    i_size = hidden_size
                    o_size = input_size

                self.cnn_trans.add_module(str(i),nn.Sequential(
                            nn.ConvTranspose1d(in_channels=o_size,out_channels=input_size,kernel_size=3,stride=2,padding=1,\
                                              output_padding=1),
                            nn.BatchNorm1d(input_size),))


            
            
        self.dense = nn.Sequential(nn.Linear(output_size_time,hidden_size),
                                     self.dense_dropout,nn.ReLU(),nn.Linear(hidden_size,2))

        self.loss_bn = nn.BatchNorm1d(1)
        if self.loss_type == "logloss":
            self.logits_dense = nn.Linear(input_size,self.encoder.embedding.weight.shape[0])

    def make_mask(self,X,valid_len):
        shape=X.shape[:2]
        if valid_len.dim() == 1:
            valid_len = valid_len.view(-1,1).repeat(1,shape[1])
        mask=(torch.arange(0,X.shape[1]).repeat(X.shape[0],1).to(X.device)<valid_len).float()
        return mask
    def forward(self,X,X_fre,valid_len,time_dis,labels,time_position_ids = None,idx_list = None,geo_position_ids = None):
        inputs,sequence_logits,final_state,X_embed,indices_list = self.encoder(X,X_fre,valid_len,time_dis,\
                                 labels,time_position_ids,idx_list,geo_position_ids)
        if self.use_cnn:
            outputs = self.cnn_trans(sequence_logits.permute(0,2,1)).permute(0,2,1)
            outputs = outputs[:,:X.shape[1]]
        else:
            outputs = sequence_logits
        mask = self.make_mask(X_embed,valid_len)
        outputs[mask==0] = 0
        X_embed[mask==0] = 0
        if self.loss_type == "mse":
            loss_val = ((outputs-X_embed)**2).sum(dim=1).sum(dim=1)/(valid_len.float()*outputs.shape[-1])
            loss = torch.mean(loss_val)
            loss_logits = self.loss_bn(loss_val.unsqueeze(-1))
            loss_logits = torch.sigmoid(loss_logits)
            logits =torch.cat([1-loss_logits,loss_logits],dim=1)
        else:
            prob = self.logits_dense(outputs)
            log_prob = F.log_softmax(prob,dim=-1)
            log_prob = log_prob.gather(dim=-1,index = X.unsqueeze(-1)).squeeze(-1)
            log_prob[mask==0] = 0
            loss = -log_prob.sum()/mask.sum()
            loss_logits = torch.mean(torch.exp(log_prob),dim=-1)
            logits =torch.cat([loss_logits.unsqueeze(-1),1-loss_logits.unsqueeze(-1)],dim=1)
        return  logits,loss,inputs
