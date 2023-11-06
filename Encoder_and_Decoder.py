# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 09:36:26 2023

@author: adina
"""
import torch
from torch import nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
"""Encoding and Decoding in the NMT Model"""
def verbose_attention(encoder_state_vectors, query_vector):
    """A descriptive version of the neural attention mechanism 
    
    Args:
        encoder_state_vectors (torch.Tensor): 3dim tensor from bi-GRU
          in encoder
        query_vector (torch.Tensor): hidden state in decoder GRU 
    """
    # getting the shape
    batch_size, num_vectors, vector_size = encoder_state_vectors.size()
    # getting the scores using dot product 
    vector_scores = \
        torch.sum(encoder_state_vectors * query_vector.view(batch_size,1,vector_size),
                  dim=2)
    # all the probabilities (scalars) foreach vector using the dot product
    vector_probabilities = F.softmax(vector_scores,dim=1)
    # calculating the weights foreach vector 
    weighted_vectors = \
    encoder_state_vectors*vector_probabilities.view(batch_size,num_vectors,1)
    # summing all the vectors into 1
    context_vectors = torch.sum(weighted_vectors,dim=1)
    
    return context_vectors, vector_probabilities , vector_scores


def terse_attention(encoder_state_vectors, query_vector):
    """A shorter and more optimized version of the neural attention mechanism
    
    Args:
        encoder_state_vectors (torch.Tensor): 3dim tensor from bi-GRU in encoder
        query_vector (torch.Tensor): hidden state
    """
    vector_scores = torch.matmul(encoder_state_vectors,query_vector.unsqueeze(dim=2).squeeze())
    vector_probabilities = F.softmax(vector_scores , dim=1)
    context_vectors = torch.matmul(encoder_state_vectors.transpose(-2,-1),
                                   vector_probabilities.unsqueeze(dim=2).squeeze())
    return context_vectors, vector_probabilities



class NMTEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size):
        """Args:
            num_embeddings (int): number of embeddings is the size of source vocabulary
            embedding_size (int): size of the embedding vectors
            rnn_hidden_size (int): size of the RNN hidden state vectors
        """
        super(NMTEncoder,self).__init__()
        
        self.source_embedding = nn.Embedding(num_embeddings =num_embeddings ,
                                             embedding_dim = embedding_size,
                                             padding_idx=0)
        # The bidirectional rnn layer 
        self.birnn =nn.GRU(input_size = embedding_size, 
                           hidden_size = rnn_hidden_size,
                           bidirectional=True,
                           batch_first= True)
    
    def forward(self, x_source, x_lengths):
       """The forward pass of the model
       
       Args:
           x_source (torch.Tensor): the input data tensor.
               x_source.shape is (batch, seq_size)
           x_lengths (torch.Tensor): a vector of lengths for each item in
           the batch
       Returns:
           a tuple: x_unpacked (torch.Tensor), x_birnn_h (torch.Tensor)
               x_unpacked.shape = (batch, seq_size, rnn_hidden_size * 2)
               x_birnn_h.shape = (batch, rnn_hidden_size * 2)
       """
       x_embedded = self.source_embedding(x_source)
        # create PackedSequence; x_packed.data.shape=(number_items, embeddign_size)
       x_packed = pack_padded_sequence(x_embedded, x_lengths.detach().cpu().numpy(), 
                                        batch_first=True)
        
        # x_birnn_h.shape = (num_rnn, batch_size, feature_size)
       x_birnn_out, x_birnn_h  = self.birnn(x_packed)
        # permute to (batch_size, num_rnn, feature_size)
       x_birnn_h = x_birnn_h.permute(1, 0, 2)
        
        # flatten features; reshape to (batch_size, num_rnn * feature_size)
        #  (recall: -1 takes the remaining positions, 
        #           flattening the two RNN hidden vectors into 1)
       x_birnn_h = x_birnn_h.contiguous().view(x_birnn_h.size(0), -1)
        
       x_unpacked, _ = pad_packed_sequence(x_birnn_out, batch_first=True)
        
       return x_unpacked, x_birnn_h




class NMTDecoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size,
                 bos_index):
        """Args: 
            num_embeddings (int): number of embeddings is also the number of 
               unique words in target vocabulary
            embedding_size (int): the embedding vector size
            rnn_hidden_size (int): size of the hidden rnn state
            bos_index(int): begin-of-sequence index
        """
        super(NMTDecoder, self).__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self.target_embedding = nn.Embedding(num_embeddings = num_embeddings,
                                             embedding_dim = embedding_size)
        
        self.gru_cell = nn.GRUCell(input_size = embedding_size + rnn_hidden_size, 
                                   hidden_size = rnn_hidden_size)
        
        self.hidden_map = nn.Linear(rnn_hidden_size , rnn_hidden_size)
        self.classifier = nn.Linear(rnn_hidden_size  * 2 , num_embeddings)
        self.bos_index = bos_index
        self._sampling_temperature = 3 
        
    def _init_indices(self, batch_size):
        """ return the BEGIN-OF-SEQUENCE index vector """
        return torch.ones(batch_size, dtype=torch.int64) * self.bos_index
    
    def _init_context_vectors(self, batch_size):
        """ return a zeros vector for initializing the context """
        return torch.zeros(batch_size, self._rnn_hidden_size)
            
    def forward(self, encoder_state, initial_hidden_state, target_sequence, 
                sample_probability=0.0):
        """The forward pass of the model.
        Args: 
            encoder_state(torch tensor) : the output of the NMTEncoder
            initial_hidden_state (torch tensor) : The last hidden state 
                in the  NMTEncoder
            target_sequence (torch tensor) :  the target text data tensor
            sample_probability (float):the schedule sampling parameter
                probabilty of using model's predictions at each decoder step
        Returns : 
            output_vectors (torch.Tensor): prediction vectors at
                each output step
        """
        if target_sequence is None:
            sample_probability = 1.0
        else :
            # We are making an assumption there: The batch is on first
           # The input is (Batch, Seq)
           # We want to iterate over sequence so we permute it to (S, B)
           target_sequence = target_sequence.permute(1, 0)
           output_sequence_size = target_sequence.size(0)
        # use the provided encoder hidden state as the initial hidden state
        h_t = self.hidden_map(initial_hidden_state)
        batch_size = encoder_state.size(0)
        
         # initialize context vectors to zeros
        context_vectors = self._init_context_vectors(batch_size)
       
        # initialize first y_t word as BOS
        y_t_index = self._init_indices(batch_size)
        
        h_t = h_t.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)
        
        output_vectors = []
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()
        
        for i in range(output_sequence_size):
            
            """if the random sample is smaller than the sample probability, 
            it uses the model’s predictions during that iteration:
            """
            use_sample = np.random.random() < sample_probability
            if not use_sample:
                y_t_index = target_sequence[i]
            
            # Step 1: Embed word and concat with previous context
            y_input_vector = self.target_embedding(y_t_index)
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)
           
            # Step 2: Make a GRU step, getting a new hidden vector
            h_t = self.gru_cell(rnn_input, h_t)
            self._cached_ht.append(h_t.cpu().detach().numpy())
            
            # Step 3: Use the current hidden to attend to the encoder state
            context_vectors, p_attn, _ = verbose_attention(encoder_state_vectors=encoder_state, 
                                                           query_vector=h_t)
            
            # auxillary: cache the attention probabilities for visualization
            self._cached_p_attn.append(p_attn.cpu().detach().numpy())
            
            # Step 4: Use the current hidden and context vectors 
            # to make a prediction to the next word
            prediction_vector = torch.cat((context_vectors, h_t), dim=1)
            score_for_y_t_index = self.classifier(F.dropout(prediction_vector, 0.3))
            
            """LEARNING TO SEARCH AND SCHEDULED SAMPLING APPROACH
            we allowing the model to use its own predictions during training.
            """
            if use_sample:
                p_y_t_index = F.softmax(score_for_y_t_index * self._sampling_temperature, dim=1)
                y_t_index = torch.multinomial(p_y_t_index, 1).squeeze()
            # auxillary: collect the prediction scores
            output_vectors.append(score_for_y_t_index)
            
        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)
        
        return output_vectors



"""The NMTModel"""
class NMTModel(nn.Module):
    """ A Neural Machine Translation Model """ 
    
    def __init__(self, source_vocab_size, source_embedding_size,
    target_vocab_size, target_embedding_size, encoding_size,
    target_bos_index):
        """
        Args:
        source_vocab_size (int): number of unique words in source language
        source_embedding_size (int): size of the source embedding vectors
        target_vocab_size (int): number of unique words in target language
        target_embedding_size (int): size of the target embedding vectors
        encoding_size (int): size of the encoder RNN
        target_bos_index (int): index for BEGIN­OF­SEQUENCE token
        """
        super(NMTModel, self).__init__()
        # The encoding layer 
        self.encoder = NMTEncoder(num_embeddings=source_vocab_size,
        embedding_size=source_embedding_size,
        rnn_hidden_size=encoding_size)
        
        # The decoding layer
        decoding_size = encoding_size * 2
        self.decoder = NMTDecoder(num_embeddings=target_vocab_size,
        embedding_size=target_embedding_size,
        rnn_hidden_size=decoding_size,
        bos_index=target_bos_index)
        
    def forward(self, x_source, x_source_lengths, target_sequence, 
                sample_probability=0.0):
        """The forward pass of the model
        Args : 
            x_source (torch.Tensor): the source text data tensor.
                x_source.shape should be (batch, vectorizer.max_source_length)
             x_source_lengths torch.Tensor): the length of the sequences in x_source 
             target_sequence (torch.Tensor): the target text data tensor
             sample_probability (float): the schedule sampling parameter
               probabilty of using model's predictions at each decoder step
         Returns:
            decoded_states (torch.Tensor): 
                prediction vectors at each output step
        """
        # getting the encoder results 
        encoder_state, final_hidden_states = self.encoder(x_source, x_source_lengths)
        
        decoded_states = self.decoder(encoder_state = encoder_state,
                                      initial_hidden_state = final_hidden_states,
                                      target_sequence = target_sequence,
                                      sample_probability = sample_probability)
        return decoded_states


























