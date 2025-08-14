import numpy as np
import torch.nn as nn

from typing import Optional
from torch.nn.functional import softmax as F_softmax
import torch


from torch import (
    cat as torch_cat,
    Tensor,
    LongTensor,
)

from tools.utils import fix_random_seed

from ..layers import TokenEmbedding


__all__ = (
    'AdvancedItemEncoder',
)


class AdvancedItemEncoder(nn.Module):

    def __init__(self,
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
                # patch_len: int = 3,
                # stride: int = 3,
                # padding_patch: str = 'end'
                 ):
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

        # set seed
        if random_seed is not None:
            fix_random_seed(random_seed)

        # 0: padding token
        # 1 ~ V: item tokens
        # V+1 ~: unknown token (convert to V+1 after mask)
        if num_known_item is None:
            self.vocab_size = num_items + 1
        else:
            self.vocab_size = num_known_item + 2

        # ifeature cache
        self.ifeature_cache = nn.Embedding.from_pretrained(Tensor(ifeatures), freeze=True)

        # embedding layers
        self.token_embedding = TokenEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=hidden_dim
        )

        # embedding layers
        self.user_embedding = TokenEmbedding(
            vocab_size=self.num_users+1,
            embedding_dim=20
        )

        # temporary patching parameters
        self.patch_len_1 = 3
        self.stride_1 = 3
        self.patch_len_2 = 15
        self.stride_2 = 15

        # main layers+
        
        self.features_encoder = nn.Linear(ifeature_dim, hidden_dim * 4)
        self.features_encoder_2 = nn.Linear(hidden_dim * 4, hidden_dim )
        #self.cxt_encoder = nn.Linear( icontext_dim, hidden_dim)
        self.cxt_encoder = nn.Linear( icontext_dim, hidden_dim)
        self.a_act = nn.LeakyReLU()
        self.c_act = nn.GELU()
        #self.ac_encoder = nn.Linear(ifeature_dim + icontext_dim, hidden_dim * 4)
        #self.item_encoder = nn.Linear(hidden_dim + hidden_dim * 4, hidden_dim)
        #self.ac_encoder = nn.Linear(((hidden_dim * 4) + 16), hidden_dim * 2)
        #self.item_encoder = nn.Linear(hidden_dim + hidden_dim * 2, hidden_dim)
        self.ac_encoder = nn.Linear(((hidden_dim * 4) + hidden_dim), hidden_dim*4 )
        self.ui_encoder = nn.Linear(((hidden_dim)*2), hidden_dim )

        self.ac_proj = nn.Linear(((hidden_dim * 4) ), hidden_dim )
        self.item_encoder = nn.Linear(hidden_dim + hidden_dim*4 , hidden_dim)
        self.ac_act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.patch_encoder_1 = nn.Linear(self.patch_len_1 * hidden_dim, hidden_dim)
        self.patch_encoder_2 = nn.Linear(self.patch_len_2 * hidden_dim, hidden_dim)

        ########################################
        # Patching
        def repeat_last_value(x, stride):
            # Get the last value along the L dimension
            last_value = x[:, -1:, :]
            # Repeat the last value along the L dimension
            repeated_last_value = last_value.repeat(1, stride, 1)
            # Concatenate the repeated last value to the original tensor
            return torch_cat([x, repeated_last_value], dim=1)

        # self.patch_len = patch_len
        # self.stride = stride
        # self.padding_patch = padding_patch
        patch_num_1 = int((sequence_len - self.patch_len_1)/self.stride_1 + 1)
        patch_num_2 = int((sequence_len - self.patch_len_2)/self.stride_2 + 1)
        # if padding_patch == 'end': # can be modified to general case - could be used to pad in the beginning
        # self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
        # self.padding_patch_layer = lambda x: repeat_last_value(x, self.stride_1)
        # patch_num += 1

        ########################################

    def forward(self,
                users: Tensor,
                tokens: LongTensor,  # (b x L|C)
                icontexts: Tensor,  # (b x L|C x d_Ci)
                item_type: str
                ):

        # get ifeatures from cache
        ifeatures = self.ifeature_cache(tokens)
        ifeatures_encoded = self.features_encoder(ifeatures)
        #ifeatures_encoded = self.a_act(ifeatures_encoded)
        #ifeatures_encoded = self.dropout1(ifeatures_encoded)

        ifeatures_encoded_2 = self.features_encoder_2(ifeatures_encoded)
        #ifeatures_encoded = self.a_act(ifeatures_encoded)
        icontexts_encoded = self.cxt_encoder(icontexts)

        #icontexts_encoded = self.c_act(icontexts_encoded)
        # get ac vector
        ac = torch_cat([ifeatures_encoded, icontexts_encoded], dim=-1)
        #ac = ifeatures_encoded
        #ac = torch_cat([ifeatures, icontexts], dim=-1)
        #ac = self.ac_act(ac)
        #ac = self.dropout(ac)
        ac_vector_comb = self.ac_encoder(ac)
        #ac_vector = self.ac_act(ac_vector)
        ac_vector = self.dropout(ac_vector_comb)
        ac_vector_proj = self.ac_proj(ac_vector_comb) 

        # get token vector
        if self.num_known_item is not None:
            unk = self.vocab_size - 1
            tokens[tokens >= unk] = unk
        token_vector = self.token_embedding(tokens)
        users_vector = self.user_embedding(users)

        #i_u = torch_cat([token_vector, torch.unsqueeze(users_vector,1).repeat(1, token_vector.size(1), 1)], dim=-1)
        #i_u = self.ui_encoder(i_u)

        # get item vector
        #print('the size of token_vector in advanced is: ', token_vector.size() )
        #print('the size of ac_vector in advanced is: ', ac_vector.size() )
        #ifeatures_weight = self.features_encoder_2(ifeatures_encoded)
        #token_vector = token_vector *  self.ac_act(ifeatures_weight)

        vector = torch_cat([token_vector, ac_vector], dim=-1)
        vector = self.item_encoder(vector) # b x L|C x d
        # multiply again by the softmax(features) to influence the embedding by the features
        # to allow items to be infleuenced by the features
        #Interval = torch.diff(icontexts, dim=1)
        #one = torch.ones(icontexts.size(0), 1, icontexts.size(2), device=icontexts.device, dtype=icontexts.dtype)
        # Concatenate
        #T_diff = torch.cat([one, Interval], dim=1)

        
        #vector = self.dropout1(vector)
        
        ########################################
        
        return vector, ifeatures_encoded_2, icontexts, users_vector
