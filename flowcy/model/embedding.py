import torch 
import torch.nn as nn

import math

# Overview
# -- unconditional: input = E_token + E_position + E_time
# -- conditional: input = E_token + E_segment + E_position + E_time
# 
# note
# - time embedding is only apply to cytokine sequence, not the receptor sequence


class token_embedding(nn.Module):
    """
    embed each token into d_model dimension vector
    """
    def __init__(self, tokens:torch.tensor, num_vocabs:int, d_model:int):
        super().__init__()
        self.tokens = tokens
        self.d_model = d_model
        self.embedding = nn.Embedding(num_vocabs, self.d_model)

    def embed(self) -> torch.tensor:
        return self.embedding(self.tokens)



class segment_embedding(nn.Module):
    """
    for conditional generation
    return zero tensor if unconditional generation

    """
    def __init__(self, tokens:torch.tensor, d_model:int, is_conditional:bool):
        super().__init__()
        self.tokens = tokens
        self.num_token = len(tokens)
        self.d_model = d_model
        self.is_conditional = is_conditional
        self.segment_embedding = nn.Embedding(2, self.d_model) 

    def embed(self) -> torch.tensor:
        if self.is_conditional:
            return self.segment_embedding(self.tokens)
        else:
            return torch.zeros((self.num_token, self.d_model))



class rotary_position_embedding(nn.Module):
    def __init__(self, num_token:int, d_model:int, query:torch.tensor, key:torch.tensor):
        super().__init__()
        self.num_token = num_token
        self.query = query
        self.key = key
        self.d_model = d_model


    def _get_frequencies(self) -> torch.tensor:
        """
        i in {1, 2, ..., d/2}
        """
        frequencies = torch.ones(self.d_model)
        token_position = torch.arange(self.num_token)
        i = torch.arange(self.d_model)
        i = torch.div(i, 2, rounding_mode="floor") - 1
        frequencies = torch.outer(token_position,  ((1e4) ** (-2 * i / self.d_model)))
        return frequencies


    def rotate(self, matrix:torch.tensor) -> torch.tensor:
        """
        rotate the query and key vector using element wise multiplication
        rotation matrix = [mθ_1, ..., mθ_{d/2}]

        frequencies: (num_token, d_model)
        key_matrix: (num_token)
        query_matrix: (num_token, d_model) result from applying query and key to x
        """
        frequencies = self._get_frequencies()

        sin = torch.sin(frequencies)
        cos = torch.cos(frequencies)

        indices = torch.arange(self.d_model)
        indices_copy = indices.clone()
        indices[0::2] = indices[1::2]
        indices[1::2] = indices_copy[0::2]
        permutated_matrix = matrix[:, indices]
        permutated_matrix[:, 0::2] *= -1

        rotated_matrix = cos * matrix + sin * permutated_matrix

        return rotated_matrix



class time_embedding(nn.Module):
    """
    sinusoidal time embedding + MLP to project time into d_model vector
    output time embedding: num_token by d_model
    """

    def __init__(self, t:torch.tensor, d_model:int, d_time:int, max_position:int):
        super().__init__()
        self.time = t
        self.d_model = d_model
        self.d_time = d_time 
        self.max_position = max_position 

        self.mlp = nn.Sequential(nn.Linear(d_model, d_time), nn.ReLU(), 
                                 nn.Linear(d_time, d_time), nn.ReLU(),
                                 nn.Linear(d_time, d_model), nn.ReLU(), 
                                 nn.LayerNorm(d_model))


    def _get_frequencies(self) -> torch.tensor:
        half_d_model = self.d_model // 2
        half_d_model_index = torch.arange(half_d_model) 
        freq = math.log(self.max_position) / (half_d_model - 1) #
        freq = torch.exp(-half_d_model_index * freq)
        return freq


    def _embed_time(self) -> torch.tensor:
        timestep = self.time * self.max_position  # allow for the model to be sensitive to tiny time change 
        angular_frequencies = self._get_frequencies() 
        sin_embed = torch.cos(angular_frequencies * timestep[:, None])
        cos_embed = torch.sin(angular_frequencies * timestep[:, None])
        sinusoidal_embedding = torch.cat((sin_embed, cos_embed), dim=0).T.flatten()

        return sinusoidal_embedding[None, :]


    def embed(self) -> torch.tensor:
        """
        project the sinsusodial embedding into high dimensional vector via MLP
        """
        embeded_time = self._embed_time()
        return self.mlp(embeded_time)