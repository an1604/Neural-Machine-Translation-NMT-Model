# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:00:48 2023

@author: adina
"""
# imports 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Vocab import Vocabulary ,SequenceVocabulary
from NMTVectorizer import NMTVectorizer
from NMTDataset import NMTDataset
import  Encoder_and_Decoder 
from Encoder_and_Decoder import NMTDecoder , NMTDecoder ,NMTModel
import pdb
"""Generating minibatches for the PackedSequence data structure"""
def generate_nmt_batches(dataset , batch_size , shuffle = True , drop_last= True , 
                         device = 'cpu'):
    """Agenerator function which wraps the PyTorch DataLoader (NMT version)"""
    try:
        dataloader = DataLoader(dataset= dataset , batch_size= batch_size ,
                                shuffle= shuffle, drop_last= drop_last)
        
        for data_dict in dataloader:
            lenghts = data_dict['x_source_length'].numpy()
            sorted_length_indices = lenghts.argsort()[::-1].tolist()
            
            out_data_dict = {}
            for name ,tensor in data_dict.items():
                out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
            
            yield out_data_dict
    except RuntimeError as e :
        print('this is the prolem {}'.format(e))

def normalize_sizes(y_pred, y_true):
    """Normalize tensor sizes
    
    Args:
        y_pred (torch.Tensor): the output of the model
            If a 3-dimensional tensor, reshapes to a matrix
        y_true (torch.Tensor): the target predictions
            If a matrix, reshapes to be a vector
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true

def compute_accuracy(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)
    
    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()
    
    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100

def sequence_loss(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)




# settings 
device = 'cpu'
learning_rate=5e-4
batch_size=32
num_epochs=100
source_embedding_size=24
target_embedding_size=24
encoding_size=32

dataset = NMTDataset.load_dataset_and_make_vectorizer('simplest_eng_fra.csv')
vectorizer = dataset.get_vectorizer()

model = NMTModel(source_vocab_size = len(vectorizer.source_vocab),
                 source_embedding_size = source_embedding_size,
                 target_vocab_size =len(vectorizer.target_vocab) , 
                 target_embedding_size =target_embedding_size ,
                 encoding_size =encoding_size , 
                 target_bos_index = vectorizer.target_vocab.begin_seq_index)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                           mode='min', factor=0.5,
                                           patience=1)
mask_index = vectorizer.target_vocab.mask_index


try:
    for epoch_index in range(num_epochs):
        # computing the sample probability
        sample_probability = (20 + epoch_index) / num_epochs
        
        # Iterate over training dataset
        # setup: batch generator, set loss and acc to 0, set train mode on
        dataset.set_split('train')
        batch_generator = generate_nmt_batches(dataset, 
                                               batch_size=batch_size, 
                                               device=device)
        running_loss = 0.0
        running_acc = 0.0
        model.train()
    
        print('Start the training...')
        for batch_index, batch_dict in enumerate(batch_generator):
            # step 1. zero the gradients
            optimizer.zero_grad()
        
            # step 2. compute the output
            y_pred = model(batch_dict['x_source'], 
                           batch_dict['x_source_length'], 
                           batch_dict['x_target'],
                           sample_probability=sample_probability)
        
            # step 3. compute the loss
            y_target = batch_dict['y_target']
            loss = sequence_loss(y_pred, y_target, mask_index)
        
            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()
        
            # compute the running loss and running accuracy
            running_loss += (loss.item() - running_loss) / (batch_index + 1)
        
            # compute the accuracy
            acc_t = compute_accuracy(y_pred, y_target, mask_index)
            running_acc += (acc_t - running_acc) / (batch_index + 1)
        
        print('Epoch {} - Train loss : {}'.format(epoch_index, running_loss))
        print('Epoch {} - Train accuracy : {}'.format(epoch_index, running_acc))
    
        # ... (validation loop)

except KeyboardInterrupt:
    print("Exiting loop")



"""Test the model"""
import NMTSampler
from NMTSampler import NMTSamplerClass
import matplotlib.pyplot as plt
model = model.eval().to(device)
sampler = NMTSamplerClass(vectorizer = vectorizer , model= model)

dataset.set_split('test')
batch_generator = generate_nmt_batches(dataset, 
                                       batch_size=batch_size, 
                                       device=device)


test_results = []

for batch_dict in batch_generator:
    sampler.apply_to_batch(batch_dict)
    for i in range(batch_size):
        test_results.append(sampler.get_ith_item(i, False))
        
# Visualizing the results 
plt.hist([r['bleu-4'] for r in test_results], bins=100);
np.mean([r['bleu-4'] for r in test_results]), np.median([r['bleu-4'] for r in test_results])
    


def get_source_sentence(vectorizer, batch_dict, index):
    indices = batch_dict['x_source'][index].cpu().data.numpy()
    vocab = vectorizer.source_vocab
    return sentence_from_indices(indices, vocab)

def get_true_sentence(vectorizer, batch_dict, index):
    return sentence_from_indices(batch_dict['y_target'].cpu().data.numpy()[index], vectorizer.target_vocab)
    
def get_sampled_sentence(vectorizer, batch_dict, index):
    y_pred = model(x_source=batch_dict['x_source'], 
                   x_source_lengths=batch_dict['x_source_length'], 
                   target_sequence=batch_dict['x_target'], 
                   sample_probability=1.0)
    return sentence_from_indices(torch.max(y_pred, dim=2)[1].cpu().data.numpy()[index], vectorizer.target_vocab)

def get_all_sentences(vectorizer, batch_dict, index):
    return {"source": get_source_sentence(vectorizer, batch_dict, index), 
            "truth": get_true_sentence(vectorizer, batch_dict, index), 
            "sampled": get_sampled_sentence(vectorizer, batch_dict, index)}
    
def sentence_from_indices(indices, vocab, strict=True):
    ignore_indices = set([vocab.mask_index, vocab.begin_seq_index, vocab.end_seq_index])
    out = []
    for index in indices:
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            return " ".join(out)
        else:
            out.append(vocab.lookup_index(index))
    return " ".join(out)

results = get_all_sentences(vectorizer, batch_dict, 1)
results
