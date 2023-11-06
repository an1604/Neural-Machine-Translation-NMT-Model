# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 09:28:49 2023

@author: adina
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
from NMTVectorizer import NMTVectorizer
from torch.nn.utils.rnn import pad_sequence
class NMTDataset(Dataset):
    def __init__(self, text_df, vectorizer):
        """
        Args:
            surname_df (pandas.DataFrame): the dataset
            vectorizer (SurnameVectorizer): vectorizer instatiated from dataset
        """
        self.text_df = text_df
        self._vectorizer = vectorizer
        
        self.train_df = self.text_df[self.text_df['split']=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.text_df[self.text_df['split']=='val']
        self.validation_size = len(self.val_df)

        self.test_df = self.text_df[self.text_df['split']=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')
        
        
    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_csv):
        """Load dataset and make a new vectorizer from scratch
        Args:
            dataset_csv (str): location of the dataset
       Returns:
           an instance of SurnameDataset
        """
        text_df = pd.read_csv(dataset_csv)
        train_subset = text_df[text_df['split']=='train']
        return cls(text_df= text_df , vectorizer= NMTVectorizer.from_dataframe(train_subset))
    
    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point: (x_data, y_target, class_index)
        """
        row = self._target_df.iloc[index]

        vector_dict = self._vectorizer.vectorize(row.source_language, row.target_language)

        return {"x_source": vector_dict["source_vector"], 
                "x_target": vector_dict["target_x_vector"],
                "y_target": vector_dict["target_y_vector"], 
                "x_source_length": vector_dict["source_length"]}
       
    def get_num_batches(self, batch_size):
       """Given a batch size, return the number of batches in the dataset
       
       Args:
           batch_size (int)
       Returns:
           number of batches in the dataset
       """
       return len(self) // batch_size



        