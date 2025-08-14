from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, LongTensor, cat as torch_cat

from tools.utils import fix_random_seed
from ..layers import TokenEmbedding

__all__ = ("ConvItemEncoder",)


class ConvItemEncoder(nn.Module):
    def __init__(
        self,
        sequence_len: int,
        num_items: int,
        num_users: int,
        ifeatures: np.ndarray,
        ifeature_dim: int,
        icontext_dim: int,
        hidden_dim: int = 64,
        num_known_item: Optional[int] = None,
        random_seed: Optional[int] = None,
        dropout_prob: Optional[float] = None,
    ) -> None:
        super().__init__()

        # data params
        self.sequence_len = sequence_len
        self.num_items = num_items
        self.num_users = num_users
        self.ifeature_dim = ifeature_dim
        self.icontext_dim = icontext_dim

        # main params
        self.hidden_dim = hidden_dim
        self.num_known_item = num_known_item

        # optional params
        self.random_seed = random_seed
        if random_seed is not None:
            fix_random_seed(random_seed)

        # vocab sizing
        # 0: padding token; 1..V: known items; V+1: unknown
        if num_known_item is None:
            self.vocab_size = num_items + 1
        else:
            self.vocab_size = num_known_item + 2

        # ifeature cache (frozen)
        self.ifeature_cache = nn.Embedding.from_pretrained(torch.Tensor(ifeatures), freeze=True)

        # embeddings
        self.token_embedding = TokenEmbedding(vocab_size=self.vocab_size, embedding_dim=hidden_dim)
        self.user_embedding = TokenEmbedding(vocab_size=self.num_users + 1, embedding_dim=20)

        # encoders
        self.features_encoder = nn.Linear(ifeature_dim, hidden_dim * 4)
        self.features_encoder_out = nn.Linear(hidden_dim * 4, hidden_dim)
        self.cxt_encoder = nn.Linear(icontext_dim, hidden_dim)

        # fusion encoders
        self.ac_encoder = nn.Linear((hidden_dim * 4) + hidden_dim, hidden_dim * 4)
        self.item_encoder = nn.Linear(hidden_dim + hidden_dim * 4, hidden_dim)

        self.dropout = nn.Dropout(p=dropout_prob)

        # # patch params kept for parity (used only to size the unused patch encoders previously)
        # self.patch_len_1 = 3
        # self.stride_1 = 3
        # self.patch_len_2 = 15
        # self.stride_2 = 15

    def forward(
        self,
        users: Tensor,
        tokens: LongTensor,  # (B, L|C)
        icontexts: Tensor,   # (B, L|C, d_Ci)
        item_type: str,
    ):
        # ifeatures from cache
        ifeatures = self.ifeature_cache(tokens)
        ifeatures_encoded = self.features_encoder(ifeatures)
        ifeatures_encoded_out = self.features_encoder_out(ifeatures_encoded)

        # icontext encoding
        icontexts_encoded = self.cxt_encoder(icontexts)

        # concatenate encoded features and context, then project
        ac = torch_cat([ifeatures_encoded, icontexts_encoded], dim=-1)
        ac_vector_comb = self.ac_encoder(ac)
        ac_vector = self.dropout(ac_vector_comb)

        # token & user embeddings
        if self.num_known_item is not None:
            unk = self.vocab_size - 1
            tokens = tokens.clone()
            tokens[tokens >= unk] = unk
        token_vector = self.token_embedding(tokens)
        users_vector = self.user_embedding(users)

        # final item representation
        vector = torch_cat([token_vector, ac_vector], dim=-1)
        vector = self.item_encoder(vector)

        return vector, ifeatures_encoded_out, icontexts, users_vector
