import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import token_embedding, segment_embedding, time_embedding
from transformer import transformer_stack
from tokenizer import VOCAB_SIZE

# Overview
# - Denoiser that combine embedding and transformer to predict the amino acid
#   logit from noisy protein sequence
# 
# - input: receptor Sequence <eos> binder sequence <eos> receptor sequence
# - output: 
#
# Architecture
# - embedding block
# - transformer block
# - MLP block: allow the denoiser to output logit


class embeddings_layer():
    def __init__(self, tokens:torch.tensor, time:float, num_vocab:int, d_model:int, d_time:int, is_conditional:bool):
        self.tokens = tokens
        self.num_vocabs = num_vocab
        self.d_model = d_model
        self.is_conditional = is_conditional

        self.token_embedding = token_embedding(tokens, num_vocab, d_model)
        self.segment_embedding = segment_embedding(tokens, d_model, is_conditional)
        self.time_embedding = time_embedding(time, d_model, d_time)

    def forward(self) -> torch.tensor:
        token_embedding = self.token_embedding.embed(self.tokens)
        segment_embedding = self.segment_embedding.embed(self.tokens)
        time_embedding = self.time_embedding.embed()
        if self.is_conditional:
            return token_embedding + segment_embedding + time_embedding
        else:
            return token_embedding + time_embedding


class output_layer():
    """
    output the predicted logit for each amino acid in the sequences
    """

    def __init__(self, d_model:int, num_vocab:int):
        self.d_model = d_model
        self.nn = nn.Sequential(nn.Linear(d_model, d_model), nn.Relu(), 
                                nn.LayerNorm(d_model), nn.Linear(d_model, num_vocab))
        
    def forward(self, input:torch.tensor) -> torch.tensor:
        return self.nn(input)



class FlowCyModel():
    def __init__(self, tokens:torch.tensor, time:float, d_model:int, d_time:int, num_layer:int, num_head:int, is_conditional):
        self.tokens = tokens
        self.embedding = embeddings_layer(tokens, time, VOCAB_SIZE, d_model, d_time, is_conditional)
        self.transformer = transformer_stack(num_layer, d_model, num_head)
        self.output_head = output_layer(d_model, VOCAB_SIZE)


    def forward(self):
        """
        embed --> transformer --> MLP
        """
        embedded_tokens = self.embedding.forward()
        transformer_output = self.transformer.forward(embedded_tokens)
        logit = self.output_head.forward(transformer_output)

        predicted_probabilites = F.softmax(logit, dim=-1)
        return predicted_probabilites



        