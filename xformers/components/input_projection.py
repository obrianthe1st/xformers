# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.init import (
    _calculate_fan_in_and_fan_out,
    _no_grad_uniform_,
    xavier_uniform_,
)

_perf_self_attention_warning = True


def small_init_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    r"""Fills the input `Tensor` with values according to the method
    described in `Transformer Without Tears`_, using a uniform distribution.

    This is a variation of the Xavier init. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + 4 * \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    .. _`Transformer Without Tears`: https://doi.org/10.5281/zenodo.3525484

    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)


@dataclass
class InProjParams:
    in_features: int
    out_features: int
    bias: bool
    small_init: bool = False


class InputProjection(nn.Module):
    """
    Handle all the input projections in one go, opportunistically fuse some operations.

    CREDITS: Inspired by https://github.com/pytorch/text/blob/master/torchtext/nn/modules/multiheadattention.py
    and the MultiHeadAttention implementation from PyTorch
    """

    def __init__(
        self,
        query_proj_params: InProjParams,
        key_proj_params: Optional[InProjParams] = None,
        value_proj_params: Optional[InProjParams] = None,
        self_attention: bool = False,
    ):

        super().__init__()

        assert not self_attention or (
            key_proj_params is None and value_proj_params is None
        ), "if using self-attention, only one parameter is valid"

        # If nothing is specified for key and value, use the same as query
        if key_proj_params is None:
            key_proj_params = query_proj_params

        if value_proj_params is None:
            value_proj_params = query_proj_params

        # Catch a beneficial case, if Q,K,V dimensions are the same
        self.self_attention = self_attention
        self.out_features = query_proj_params.out_features
        self.q_p_params = query_proj_params
        self.k_p_params = key_proj_params
        self.v_p_params = value_proj_params

        # - handle all the weights
        # save the requested init method
        if self.self_attention:
            # We can use a single weight and bias buffer, which will speed up self attention
            self.in_proj = nn.Linear(
                in_features=query_proj_params.in_features,
                out_features=3 * self.out_features,
                bias=query_proj_params.bias,
            )
        else:
            # The dimensions are different, use seperate buffers
            self.q_proj = nn.Linear(
                query_proj_params.in_features,
                self.out_features,
                bias=query_proj_params.bias,
            )
            self.k_proj = nn.Linear(
                key_proj_params.in_features,
                self.out_features,
                bias=key_proj_params.bias,
            )
            self.v_proj = nn.Linear(
                value_proj_params.in_features,
                self.out_features,
                bias=value_proj_params.bias,
            )

        # - multi-head attention specific init for the weights and biases
        self._reset_parameters()

    def _reset_parameters(self):
        if self.self_attention:
            # Init each of the projection blocks independently
            out_features = self.in_proj.weight.shape[0] // 3

            with torch.no_grad():
                self.in_proj.weight[:out_features, :] = self._init_weights(
                    self.q_p_params, self.in_proj.weight[:out_features, :]
                )
                self.in_proj.weight[
                    out_features : 2 * out_features, :
                ] = self._init_weights(
                    self.q_p_params,
                    self.in_proj.weight[out_features : 2 * out_features, :],
                )
                self.in_proj.weight[2 * out_features :, :] = self._init_weights(
                    self.q_p_params, self.in_proj.weight[2 * out_features :, :]
                )
        else:
            self.q_proj.weight = self._init_weights(self.q_p_params, self.q_proj.weight)
            self.k_proj.weight = self._init_weights(self.k_p_params, self.k_proj.weight)
            self.v_proj.weight = self._init_weights(self.v_p_params, self.v_proj.weight)

    @staticmethod
    def _init_weights(params: InProjParams, weights: torch.Tensor, gain: float = 1.0):
        if params.small_init:
            return small_init_(weights, gain=gain)
        else:
            return xavier_uniform_(weights, gain=gain)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.self_attention:
            assert self.in_proj is not None
            assert id(query) == id(key) and id(query) == id(
                value
            ), "Self attention was requested, but query, key and value differ"

            # All the inputs have the same dimension, a small optimization is to compute the projections all in one go
            qkv = self.in_proj(query)

            q, k, v = map(
                lambda x: x.contiguous(),
                qkv.split(self.out_features, dim=-1),
            )
            return q, k, v

        else:
            # Warn about a possible speed benefit if self attention and not declared as such
            global _perf_self_attention_warning

            if (
                id(query) == id(key)
                and id(query) == id(value)
                and _perf_self_attention_warning
            ):
                logging.warning(
                    "Seems that this is self-attention, but not declared as such, which makes it slightly slower"
                )
                logging.warning(
                    "Please declare this Attention block as self-attention"
                    + " if you're interested in the best possible speed."
                )
                _perf_self_attention_warning = False

            # We have a projection per input
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

            return q, k, v
