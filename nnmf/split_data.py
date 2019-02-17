#!/usr/bin/env python2.7
from __future__ import absolute_import, print_function
"""Simple utility to separate data into training, test, and validation."""
# Standard libraries
import argparse, os
# Third party modules
import pandas as pd
import numpy as np

def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()

def preprocess_data2(data):
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = []
    rows = []
    cols = []
    for row, col, rating in data:
        ratings.append(rating)
        rows.append(row)
        cols.append(col)
    return pd.DataFrame({'user':rows, 'item':cols, 'rating':ratings})[['user', 'item', 'rating']]

def load_data2(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data2(data)

def split_nnmf(input_='./data/data_train.csv', outfolder='./data/', outprefix='mov_kaggle', delimiter='\t',train_test_r=0.9, train_valid_r=0.9):
    """
    split oroginal dataset to train, valid and test sets. 
    :param input: the location of the input file
    :param outfolder: the location of the folder to output to
    :param outprefix: string to append to front of output filenames
    :param delimiter: delimiter to use when parsing/writing files
    :param train_test_r: Ratio of training data to test data
    :param train_valid_r:Ratio of training data to validation data
    
    :return:  no return(files are saved in outfolder)
    """

    input_filename = input_
    output_folder = outfolder
    output_prefix = outprefix
    delimiter = delimiter
    train_test_r = train_test_r
    train_valid_r = train_valid_r

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Read in data
    print('Reading in data')

    data = load_data2(input_)#pd.read_csv(input_filename, delimiter=delimiter, header=None,
                       # names=['user_id', 'item_id', 'rating'])
    # data.drop('timestamp', axis=1, inplace=True)

    # Shuffle
    data = (data.iloc[np.random.permutation(len(data))]).reset_index(drop=True)

    # Split data set into (all) train/test
    all_train_ratio = train_test_r
    num_all_train = int(len(data)*all_train_ratio)
    num_test = len(data)-num_all_train
    all_train_data = data.head(num_all_train)
    test_data = (data.tail(num_test)).reset_index(drop=True)

    # Split up (all) train data into train/validation
    train_ratio = train_valid_r
    num_train = int(len(all_train_data)*train_ratio)
    num_valid = len(all_train_data)-num_train
    train_data = (all_train_data.head(num_train)).reset_index(drop=True)
    valid_data = (all_train_data.tail(num_valid)).reset_index(drop=True)

    print("Data subsets:")
    print("Train: {}".format(len(train_data)))
    print("Validation: {}".format(len(valid_data)))
    print("Test: {}".format(len(test_data)))

    # Write data to file
    common = {'header': False, 'sep': delimiter, 'index': False}
    train_data.to_csv(os.path.join(output_folder, "{}.train".format(output_prefix)), **common)
    valid_data.to_csv(os.path.join(output_folder, "{}.valid".format(output_prefix)), **common)
    test_data.to_csv(os.path.join(output_folder, "{}.test".format(output_prefix)), **common)
