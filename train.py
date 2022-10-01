#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 04:45:16 2021

@author: karma
"""

import torch
from torchtext import data   
from torchtext.vocab import GloVe

import torch.optim as optim
import torch.nn as nn

from modelling import LSTMClassifier

from pathlib import Path

from ray import tune

from preprocessing import preprocess

import os
import warnings
warnings.filterwarnings("ignore",category=UserWarning, module="torchtext")
warnings.filterwarnings("ignore",category=UserWarning, module="torch")

import logging
logging.getLogger(tune.trainable.__name__).disabled = True
logging.getLogger(tune.utils.util.__name__).disabled = True

from settings import Settings
setting = Settings()
torch.manual_seed(setting.general["seed"])

class Trainer(tune.Trainable):
            
    def load_data(self):
        """Load training and validation data from directory datasets/processed.
        Fields are initialized and spacy tokenizers are used to tokenize read
        data. Vocabulary is built while loading data and available in Field
        objects as attributes.

        Returns:
            TabularDataset: tokenized data stored as attributes in object of 
            TabularDataset class
        """
        if not self.dataset_dir.exists():
            print(" Preprocessing Dataset..")
            preprocess()
            print(" Preprocessing Dataset.. Done.")

        self.TEXT = data.Field(tokenize="spacy",
                               batch_first=True,
                               tokenizer_language="en_core_web_sm",
                               include_lengths=True, )
        self.LABELID = data.LabelField(dtype = torch.long,
                                       sequential=False,
                                       use_vocab=False,
                                       pad_token=None,
                                       unk_token=None)
        
        fields = [('labelid', self.LABELID), ('news', self.TEXT)]
        
        #loading custom dataset
        self.train_data=data.TabularDataset(path="/".join([self.dataset_dir.as_posix(), "train.csv"]),
                                          format = 'csv',
                                          fields = fields)
        self.valid_data=data.TabularDataset(path="/".join([self.dataset_dir.as_posix(), "validation.csv"]),
                                          format = 'csv',
                                          fields = fields)
        
        return self.train_data, self.valid_data
    
    def build_vocab(self, embedding_dim=100, name="6B"):
        """Load/download GloVe vector embeddings and
        built vectors for the vocabulary.

        Args:
            embedding_dim (int, optional): Embedding dimension to use. Defaults to 100.
            name (str, optional): name of GloVe embeddings build using 6B tokens. Defaults to "6B".
        """
        cache = self.project_path / "embeddings/GloVe"
        glove = GloVe(name=name, dim=embedding_dim, cache=cache.as_posix())
        self.TEXT.build_vocab(self.train_data, min_freq=3, )
        self.TEXT.vocab.load_vectors(glove)
    
    
    def get_iterators(self, bacth_size):
        """[summary]

        Args:
            bacth_size (integer): The number of examples used for 
            one iteration of Adam optimizer.

        Returns:
            tuple: A tuple of length 2 with iterators for
            training and validation data.
        """
        train_iterator, valid_iterator = data.BucketIterator.splits(
            (self.train_data, self.valid_data), 
            batch_size = bacth_size,
            sort_key = lambda x: len(x.news),
            sort_within_batch=True,
            device = self.device)
    
        return train_iterator, valid_iterator
    
    def train_model(self, iterator):
        """Train the model for an epoch. Model weights are being updated ,and, 
        loss and accuracy of training is collected.

        Args:
            iterator (BucketIterator): An iterator to iterate of batches
            of data for training.

        Returns:
            tuple: A tuple of length 2 with training loss and accuracy for one epoch
        """
        #initialize every epoch 
        epoch_loss = 0
        epoch_acc = 0
        
        #set the model in training phase
        self.model.train()
        
        self.iteration_counter = 0

        for i, batch in enumerate(iterator):
    
            #resets the gradients after every batch
            self.optimizer.zero_grad()   
    
            #retrieve text and no. of words
            text, text_lengths = batch.news 
            text, text_lengths = text.to(self.device), text_lengths.to(self.device)
            
            labelid = batch.labelid.to(self.device)
            
            #convert to 1D tensor
            predictions = self.model(text, text_lengths).squeeze()  

            #compute the loss
            loss = self.criterion(predictions, labelid)        
            
            #compute the binary accuracy
            acc = self.model_accuracy(predictions, labelid)  
            
            #backpropage the loss and compute the gradients
            loss.backward()       
            
            #update the weights
            self.optimizer.step()      
            
            #loss and accuracy
            epoch_loss += loss.item()  
            epoch_acc += acc.item()    

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate_model(self, iterator):
        """Validate the model for an epoch on validation set. The validation 
        loss and accuracy are collected.

        Args:
            iterator (BucketIterator): An iterator to iterate of batches
            of data for validation.

        Returns:
            tuple: A tuple of length 2 with validation loss and accuracy for one epoch
        """
        
        #initialize every epoch
        epoch_loss = 0
        epoch_acc = 0
    
        #deactivating dropout layers
        self.model.eval()
            
        #deactivates autograd
        with torch.no_grad():
        
            for i, batch in enumerate(iterator):
    
                #retrieve text and no. of words
                text, text_lengths = batch.news
                
                #convert to 1d tensor
                predictions = self.model(text, text_lengths).squeeze()
                
                #compute loss and accuracy
                loss = self.criterion(predictions, batch.labelid)
                acc = self.model_accuracy(predictions, batch.labelid)
                
                #keep track of loss and accuracy
                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    def model_accuracy(self, predict, y):
        """Get O-1 accuracy of the model by comparing 
        predicted values and ground truth (news category). 

        Args:
            predict (tensor): Probability for each class for every example
            y (tensor): A list of class ids. 

        Returns:
            float: The accuracy of predictions in range (0,1)
        """
        p = torch.argmax(predict, dim=1)
        true_predict = (p==y).float()
        acc = true_predict.sum()/len(true_predict)
    
        return acc

    def step(self,):
        """A implementation of abstract method from ray.tune class API.
        One logical iteration step for training, here an epoch.

        Returns:
            dict: A dictionary of metrics to be used for comparing
            trials.
        """
        self.epoch += 1
        
        #train the model
        train_loss, train_acc = self.train_model(self.train_iterator)
        
        #evaluate the model
        valid_loss, valid_acc = self.evaluate_model(self.valid_iterator)
        
        if valid_acc > self.best_valid_acc:
            self.best_valid_acc = valid_acc

        #print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        return {"loss":valid_loss, 
                "accuracy":valid_acc, 
                "epoch": self.epoch,
                "best_accuracy": self.best_valid_acc}

    def setup(self, config):
        """A implementation of abstract method from ray.tune class API.
        Function invoked at the start of an experiment.
        Initializes loading of train and validation data. 
        Creates LSTM/BiLSTM model and loads pretrained embeddings into
        embedding layer. Adam optimizer and Cross Entropy loss criterion
        are defined.

        Args:
            config ([type]): [description]
        """
        self.project_path = Path(__file__).resolve().parent
        self.dataset_dir = self.project_path / 'datasets/processed'

        self.epoch = 0
        self.best_valid_acc = 0 

        # Read csv files
        print(" Reading dataset from *.csv files ...")
        self.load_data()
        print(" Reading dataset from *.csv files ... Done.")

        # Build vocabulary
        #print(" Building Vocabulary ... ")
        self.build_vocab(config["lstm"]["embedding_dim"])
        #print(" Building Vocabulary ... Done. ")

        self.model = LSTMClassifier(
            vocab_size=len(self.TEXT.vocab), 
            embedding_dim=config["lstm"]["embedding_dim"], 
            hidden_dim=config["lstm"]["hidden_dim"], 
            num_layers=config["lstm"]["num_layers"],
            bidirectional=config["lstm"]["bidirectional"],
            dropout=config["lstm"]["dropout"])

        # Initialize the pretrained embedding
        self.model.embedding.from_pretrained(self.TEXT.vocab.vectors)

        # Setting up GPU devices
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"
            # if torch.cuda.device_count() > 1:
            #     self.device = "cuda:1"
            #     ids = list(range(1, torch.cuda.device_count()))
            #     print("GPUs at DataParallel", ids)
            #     self.model = nn.DataParallel(self.model, device_ids=ids)
        
        #print(" Device in use : ", self.device)
        self.model = self.model.to(self.device)

        self.train_iterator, self.valid_iterator = self.get_iterators(config["batch_size"])

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr= config["learning_rate"])
        
    def save_checkpoint(self, checkpoint_dir):
        """A implementation of abstract method from ray.tune class API
        to checkpoint model at any epoch of a trial by saving
        model.state_dict() and optimizer.state_dict().

        Args:
            checkpoint_dir (str): A directory for checkpointing,
            here models/

        Returns:
            str: The directory used for checkpointing.
        """
        #with tune.checkpoint_dir(self.epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)

        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        """A implementation of abstract method from ray.tune class API
        to load model checkpoint at any epoch of a trial by loading
        model.state_dict() and optimizer.state_dict().

        Args:
            checkpoint_dir (str): A directory for checkpoints.
        """
        model_state, optimizer_state = torch.load(
        os.path.join(checkpoint_dir, "checkpoint"))

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        
    def reset_config(self, new_config):
        """To reuse the same Trainable Python process and object for multiple 
        trials. An implementation of abstract method from ray.tune class API 
        to avoid repeated overheads, eg: loading training and validation data.

        Args:
            new_config ([type]): [description]

        Returns:
            boolean: Indicated successful configuration reset.
        """
        self.epoch = 0
        self.best_valid_acc = 0

        # Build vocabulary
        #print(" Building Vocabulary ... ")
        self.build_vocab(new_config["lstm"]["embedding_dim"])
        #print(" Building Vocabulary ... Done. ")

        self.model = LSTMClassifier(
            vocab_size=len(self.TEXT.vocab), 
            embedding_dim=new_config["lstm"]["embedding_dim"], 
            hidden_dim=new_config["lstm"]["hidden_dim"], 
            num_layers=new_config["lstm"]["num_layers"],
            bidirectional=new_config["lstm"]["bidirectional"],
            dropout=new_config["lstm"]["dropout"])

        # Initialize the pretrained embedding
        self.model.embedding.from_pretrained(self.TEXT.vocab.vectors)

        # Setting up GPU devices
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"
            # if torch.cuda.device_count() > 1:
            #     self.device = "cuda:1"
            #     ids = list(range(1, torch.cuda.device_count()))
            #     print("GPUs at DataParallel", ids)
            #     self.model = nn.DataParallel(self.model, device_ids=ids)

        #print(" Device in use : ", self.device)
        self.model = self.model.to(self.device)

        self.train_iterator, self.valid_iterator = self.get_iterators(new_config["batch_size"])

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr= new_config["learning_rate"])

        return True
