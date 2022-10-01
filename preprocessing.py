#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 18:50:00 2021

@author: karma
"""
import tarfile
import pandas
import re
import html
from sklearn.model_selection import train_test_split

from pathlib import Path

from settings import Settings

class AGNewsPreprocessor:
    
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.ext = ".".join(source.split("/")[-1].split(".")[1:])
        
        self.text_field = TextField()
        self.labelid_field = LabelField()
    
    def load_csv(self, train=True, test=True, validation=False):
        """
        

        Parameters
        ----------
        train : boolean, optional
            If train.csv exists in source. The default is True.
        test : boolean, optional
            If test.csv exists in source. The default is True.
        validation : boolean, optional
            If validation.csv exists in source. The default is False.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        """
        self.train_flag = train
        self.test_flag = test
        self.valid_flag = validation
        
        data = {}
        
        if self.ext == "tar.gz":
            with tarfile.open(self.source, mode="r:gz") as tar_gz_file:
                #  Extract the file
                tar_gz_file.extractall(self.destination)
                
                for file in tar_gz_file: 
                    if file.name.endswith(".csv"):
                        if file.name.split("/")[-1].startswith("train") and self.train_flag:
                            df = pandas.read_csv("/".join([self.destination, file.name]),
                                                 names=["labelid",
                                                        "title",
                                                        "description"],
                                                 converters={
                                                     "labelid": self.labelid_field,
                                                     "title":self.text_field,
                                                     "description":self.text_field})
                            self.redefine_features(df)
                            data["train"] = AGNewsDataframe(df)

                        if file.name.split("/")[-1].startswith("test") and self.test_flag:
                            df = pandas.read_csv("/".join([self.destination, file.name]),
                                                 names=["labelid",
                                                        "title",
                                                        "description"],
                                                 converters={
                                                     "labelid": self.labelid_field,
                                                     "title":self.text_field,
                                                     "description":self.text_field})
                            self.redefine_features(df)
                            data["test"] = AGNewsDataframe(df)

                        if file.name.split("/")[-1].startswith("validation") and self.valid_flag:
                            df = pandas.read_csv("/".join([self.destination, file.name]),
                                                 names=["labelid",
                                                        "title",
                                                        "description"],
                                                 converters={
                                                     "labelid": self.labelid_field,
                                                     "title":self.text_field,
                                                     "description":self.text_field})
                            self.redefine_features(df)
                            data["validation"] = AGNewsDataframe(df)
        return data
    
    def redefine_features(self, dataframe, inplace=True):
        """
        Combine text fields 'title' and 'description' to a single field 'news'

        Parameters
        ----------
        dataframe : pandas dataframe
            dataframe with fields 'title' and 'description'
        
        inplace : boolean, optional
            If False, return a new dataframe with changes ond passed arguement
            unaffected. The default is True, ie passed arguement changed
        Returns
        -------
        dataframe (if inplace = False) or None

        """
        # Combine 'title' and 'description' colums into new column 'news',
        # then remove 'title' and 'descripttion' columns
        if inplace:
            dataframe["news"] = dataframe["title"] + ". " + dataframe["description"]
            dataframe.drop(columns=["title", "description"], inplace=inplace)
        else:
            dataframe = dataframe.copy()
            dataframe["news"] = dataframe["title"] + ". " + dataframe["description"]
            return dataframe.drop(columns=["title", "description"], inplace=inplace)

class AGNewsDataframe:
    
    def __init__(self, dataframe):
        self.data = dataframe
        self.labels = {
            0:'World',
            1:'Sports',
            2:'Business', 
            3:'Science-Technology'} 

    def split_data(self, split_ratio=0.80, random_state=None, shuffle=True,
                               stratify=None):
        """
        

        Parameters
        ----------
        train_valid_ratio : float, optional
            The ratio of length of split1 after split and total length before split.
            The default is 0.80.
        random_state: int, RandomState instance or None, default=None
            Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
        shuffle : bool, default=True
            Whether or not to shuffle the data before splitting. If shuffle=False 
            then stratify must be None.
        stratify : array-like, default=None  
            If not None, data is split in a stratified fashion, using this as 
            the class labels.
        Returns
        -------
        None.

        """        
        data0, data1 = train_test_split(self.data,train_size=split_ratio,
                                        random_state=random_state,
                                        shuffle=shuffle,
                                        stratify=stratify)
        return AGNewsDataframe(data0), AGNewsDataframe(data1)

    def save_to_csv(self, destination, index=False, header=False):
        """
        Save the dataframes to csv files 'train.csv', 'validation.csv' 
        and 'test.csv' 
        
        destination : str
            File path including filename and extension('.csv')
        index : boolean, optional
            Flag to whether add row index in csv .file
            The default is False
        header : boolean, optional
            Flag to whether add column headers in csv .file
            The default is False
        Returns
        -------
        None.

        """
        self.destination = destination
        
        # Write data into .csv files  
        self.data.to_csv(self.destination, index = index, header=header)


class TextField:
    
    def __init__(self, html="", parentheses="", url="[url]"):
        """
        Parameters
        ----------
        html : str or None, optional
            DESCRIPTION. Tag to replace HTML tags. If None, HTML tags retained.
            The default is empty string "".
        paranthesis : str or None, optional
            DESCRIPTION. Tag to replace parthesis(with its contents). If None,
            parathesis retained.
            The default is empty string "".
        url_tag : str or None, optional
            DESCRIPTION. Tag to replace URLs. If None, URLs retained.
            The default is "[url]".

        """
        self.html = html
        self.url = url
        self.parentheses = parentheses
    
    def __call__(self, text):
        return self.converter(text)
        
    def converter(self, text):
        """
        Parameters
        ----------
        text : str
            The text in a fields 'title' or 'description'.
    
        Returns
        -------
        text : str
            Processed text after all or some of below listed transformation
            -- Fixed faulty html code referencing
            -- Fixed \$ entity
            -- Handled \ character
            -- Removed multiple spaces
            -- Tranformed html entity name/number to text symbols
            -- Removed HTML tags
            -- Replaced urls with '[url]' tag
            -- Removed spaces at the beginning of sentence.
        """
        # fix faulty html code referencing
        text = re.sub(r'\s(#\d+;)', r'&\1', text)
        text = re.sub(r'\s(\w+);', r'&\1', text)
        
        # Fix '\$' eg: 'the\$' -> 'the $','AUD\$' -> 'AUD$', ' \$'-> '$'
        text = re.sub(r'([a-z])\\(\$)', r'\1 \2', text)
        text = re.sub(r'([A-Z])?\\(\$)', r'\1\2', text)
        
        # Fix '\' eg: '.\\You', 'are\of'
        #text = re.sub(r'\\\\','', text)
        text = re.sub(r'\\', " ", text)
        
        # reduce multiple white spaces to single white space
        text = re.sub(r'\s{2,}',' ', text)
        
        # html entities to text
        text = html.unescape(text)
        
        # HTML tags removed based on flag attribute html
        if self.html is not None:
            text = re.sub(r'\<.*?\>', self.html, text)
        
        # brackets/paranthesis and content in them replaced with self.parentheses
        if self.parentheses is not None:
            text = re.sub(r'\(.*\)', self.parentheses, text)
        
        # Add single white space before every sentence to avoid conflict with
        #Another regex for the same purpose "([a-z]+)\.([A-Z]+[a-z]*)"
        text = re.sub(r'([a-z]+)\.(([A-Z][a-z]*)|([A-Z]+))',r'\1. \2', text)
        
        # URLs replaced with tag in self.url
        if self.url is not None:
            text = re.sub(r'(http[s]?://)?(www\.)?\S*\w\.([A-Z]{2,}|[a-z]{2,})(/[\S]?)?',
                          self.url, text)
        
        # To remove sources and locations at start of a description
        # text = re.sub(r'^(\s*[A-Z0-9a-z]*[,]*\s*[A-Z]+[a-z0-9]*\s+[\:]+\s*)+|^([\-]+\s)',
        #               r'', text)
        
        # Remove white spaces at the beginning of a field
        text = re.sub(r'^\s+', r'', text)
        
        return text
    

class LabelField:
    def __init__(self, dtype=int, f1to0=True):
        """
        
        Parameters
        ----------
        dtype : Type function object eg int, float or str, optional
            For type casting label
        f1to0 : boolean, optional
            To convert 1-based integer indexing of labels to 0-based indexing. 
            If number of classes are c, then 1-based indexing is 1..c and 
            0-based indexing is from 0..(c-1). Only valid if self.dtype is int,
            else ignored. The default is True.
            
        """
        self.dtype = dtype
        self.f1to0 = f1to0

        
    def __call__(self, label):
        return self.converter(label)
    
    def converter(self,  label):
        """
        
        Parameters
        ----------
        labelid : object
            A target label of some type.

        Returns
        -------
        object
            Processed label of type dtype.

        """

        label = self.dtype(label)
        if isinstance(label, int):
            if self.f1to0:
                return label - 1
            else:        
                return label
    

def preprocess():
    setting = Settings()

    project_path = Path(__file__).resolve().parent
    source_path = project_path / 'datasets/raw/AG news.tar.gz'
    extract_path = project_path / 'datasets/raw'
    processed_path = project_path / 'datasets/processed'

    agnews = AGNewsPreprocessor(source_path.as_posix(), extract_path.as_posix())
    data = agnews.load_csv()

    train = data["train"]
    test = data["test"]
    
    train, valid = train.split_data(random_state=setting.general["seed"], stratify=train.data["labelid"])

    save_destination = "datasets/processed"

    processed_path.mkdir(parents=True, exist_ok=True)

    train.save_to_csv("/".join([processed_path.as_posix(),"train.csv"]))
    test.save_to_csv("/".join([processed_path.as_posix(),"test.csv"]))
    valid.save_to_csv("/".join([processed_path.as_posix(),"validation.csv"]))

if __name__ == "__main__":
    preprocess()
