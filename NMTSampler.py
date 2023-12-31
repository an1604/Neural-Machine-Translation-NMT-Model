# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:40:46 2023

@author: adina
"""
import numpy as np
import torch
from nltk.translate import bleu_score
import seaborn as sns
import matplotlib.pyplot as plt

chencherry = bleu_score.SmoothingFunction()
# Convertion function


def sentence_from_indices(indices, vocab, strict=True, return_string=True):
    ignore_indices = set(
        [vocab.mask_index, vocab.begin_seq_index, vocab.end_seq_index])
    out = []
    for index in indices:
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            break
        else:
            out.append(vocab.lookup_index(index))
    if return_string:
        return " ".join(out)
    else:
        return out


class NMTSamplerClass:

    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model

    def apply_to_batch(self, batch_dict):
        self._last_batch = batch_dict
        # Getting the prediction from the last batch
        y_pred = self.model(x_source=batch_dict['x_source'],
                            x_source_lengths=batch_dict['x_source_length'],
                            target_sequence=batch_dict['x_target'])
        # Svaing the prediction in the dictionary
        self._last_batch['y_pred'] = y_pred

        #Computing the attention probability for this batch
        attention_batched = np.stack(
            self.model.decoder._cached_p_attn).transpose(1, 0, 2)
        self._last_batch['attention'] = attention_batched

    def get_true_sentence(vectorizer, batch_dict, index):
        return sentence_from_indices(batch_dict['y_target'].cpu().data.numpy()[index], vectorizer.target_vocab)        
    
    def _get_source_sentence(self, index, return_string=True):
        indices = self._last_batch['x_source'][index].cpu().detach().numpy()
        vocab = self.vectorizer.source_vocab
        return sentence_from_indices(indices, vocab, return_string=return_string)

    def _get_reference_sentence(self, index, return_string=True):
        indices = self._last_batch['y_target'][index].cpu().detach().numpy()
        vocab = self.vectorizer.target_vocab
        return sentence_from_indices(indices, vocab, return_string=return_string)

    def _get_sampled_sentence(self, index, return_string=True):
        _, all_indices = torch.max(self._last_batch['y_pred'], dim=2)
        sentence_indices = all_indices[index].cpu().detach().numpy()
        vocab = self.vectorizer.target_vocab
        return sentence_from_indices(sentence_indices, vocab, return_string=return_string)

    def get_ith_item(self, index,batch_dict ,return_string=True):
        output = {"source": self._get_source_sentence(index, return_string=return_string),
                  "reference": self._get_reference_sentence(index, return_string=return_string),
                  "sampled": self._get_sampled_sentence(index, return_string=return_string),
                  "attention": self._last_batch['attention'][index]
                 }

        reference = output['reference']
        hypothesis = output['sampled']

        if not return_string:
            reference = " ".join(reference)
            hypothesis = " " .join(hypothesis)

        output['bleu-4'] = bleu_score.sentence_bleu(references=[reference],
                                                    hypothesis=hypothesis,
                                                    smoothing_function=chencherry.method1)
        return output
    
    # def get_all_sentences(vectorizer, batch_dict, index):
    #     return {"source": get_source_sentence(vectorizer, batch_dict, index), 
    #         "truth": get_true_sentence(vectorizer, batch_dict, index), 
    #         "sampled": get_sampled_sentence(vectorizer, batch_dict, index)}

