"""Relation Module Definition."""
from __future__ import absolute_import

import math
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import nd as F

class MultiHeadAttention(gluon.Block):
    r"""Human-object Relation Module.

    Parameters
    ----------
    num_feat: int, default is 1024
        Dimension number used in fc layers.
    num_group : int, default is 16
        Relation group number.
        dk = num_feat / num_group.
    """
    def __init__(self, num_feat=1024, num_group=16, additional_output=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_feat = num_feat
        self.num_group = num_group
        self.dim_k = int(num_feat / num_group)
        self.additional_output = additional_output
        weight_initializer = mx.init.Normal(0.01)
        with self.name_scope():
            self.to_keys     = nn.Dense(self.dim_k,    weight_initializer=weight_initializer)
            self.to_values   = nn.Dense(self.dim_k,    weight_initializer=weight_initializer)
            self.to_queries  = nn.Dense(self.dim_k,    weight_initializer=weight_initializer)
            self.unify_heads = nn.Dense(self.num_feat, weight_initializer=weight_initializer)

    # pylint: disable=arguments-differ
    def forward(self, x):
        """Forward Relation Module.

        Parameters
        ----------
        feat : mxnet.nd.NDArray or mxnet.symbol
            (M, 1024) Feature tensor (used to compute q).
        ctx_feat : mxnet.nd.NDArray or mxnet.symbol
            (N, 1024)Contextual Feature tensor (used to compute k,v).
        box: mxnet.nd.NDArray or mxnet.symbol
            (M, 4) boxes with corner encoding.
        ctx_box: mxnet.nd.NDArray or mxnet.symbol
            (N, 4) boxes with corner encoding.

        Returns
        -------
        gt_relation_feat, ctx_relation_feat
            (M, 1024).
        """ 
        e = self.dim_k                                                           # e = 1024    (feature size)
        k, v, q = x.shape[0], x.shape[0], x.shape[0]                             # k, v, q = N (number of bounding boxes)
        h = self.num_group                                                       # h = 16      (Number of groups or num_group for multi head attention)
        
        x = x.reshape(k, h, e)
        x = x.reshape(k*h, e)
        keys    = self.to_keys(x)   .reshape(k,h,e).transpose(axes=(1,0,2))    # keys    : (h, k, e)
        values  = self.to_values(x) .reshape(k,h,e).transpose(axes=(1,0,2))    # values  : (h, v, e)
        queries = self.to_queries(x).reshape(k,h,e).transpose(axes=(1,0,2))    # queries : (h, q, e)

        keys    = keys    / (self.num_feat ** (1 / 4))
        queries = queries / (self.num_feat ** (1 / 4))
        dot = F.batch_dot(lhs=queries, rhs=keys, transpose_a=False, transpose_b=True) # dot : (h, q, k)

        attention = F.softmax(dot, axis=2)

        out = F.batch_dot(lhs=attention, rhs=values, transpose_a=False, transpose_b=False) # out : (h, q, e)
        out = out.transpose(axes=(1,0,2))                                        # out : (q, h, e)
        out = out.reshape(q, -1)                                                 # out : (q, h*e)

        out = self.unify_heads(out)                                              # out : (q, e)
        return out

class EncoderLayer(gluon.Block):
    def __init__(
        self,
        num_feat = 1024, 
        num_group = 16,
        dropout = 0,
        forward_expansion = 4,
        additional_output = False,
        **kwargs
        ):
        super(EncoderLayer, self).__init__(**kwargs)
        self.num_feat  = num_feat
        self.num_group = num_group
        self.dropout = dropout
        self.forward_expansion = forward_expansion
        weight_initializer = mx.init.Normal(0.01)

        self.attention = MultiHeadAttention(num_feat=1024, num_group=16, additional_output=additional_output)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm() 
        self.dropout_layer = nn.Dropout(self.dropout)

        self.feed_forward = nn.Sequential()
        self.feed_forward.add(nn.Dense(forward_expansion * num_feat, weight_initializer=weight_initializer))
        self.feed_forward.add(nn.Activation('relu'))
        self.feed_forward.add(nn.Dense(num_feat, weight_initializer=weight_initializer))
    def forward(
        self,
        x
    ):
        attended  = self.attention(x)
        out       = self.dropout_layer(self.norm1(attended  + x))
        forwarded = self.feed_forward(out)
        out       = self.dropout_layer(self.norm2(forwarded + out))
        return out

