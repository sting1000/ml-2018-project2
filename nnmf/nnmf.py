#!/usr/bin/env python2.7
from __future__ import absolute_import, print_function
"""Trains/evaluates NNMF models."""
# Standard modules
import argparse, json, time, os
# Third party modules
import tensorflow as tf
import pandas as pd
import numpy as np
# Package modules
from nnmf.models import NNMF
from nnmf.utils import chunk_df



def load_data(train_filename, valid_filename, test_filename, delimiter='\t', col_names=['user_id', 'item_id', 'rating']):
    """Helper function to load in/preprocess dataframes"""
    train_data = pd.read_csv(train_filename, delimiter=delimiter, header=None, names=col_names)
    train_data['user_id'] = train_data['user_id'] - 1
    train_data['item_id'] = train_data['item_id'] - 1
    valid_data = pd.read_csv(valid_filename, delimiter=delimiter, header=None, names=col_names)
    valid_data['user_id'] = valid_data['user_id'] - 1
    valid_data['item_id'] = valid_data['item_id'] - 1
    test_data = pd.read_csv(test_filename, delimiter=delimiter, header=None, names=col_names)
    test_data['user_id'] = test_data['user_id'] - 1
    test_data['item_id'] = test_data['item_id'] - 1

    return train_data, valid_data, test_data

def train(model, sess, saver, train_data, valid_data, batch_size, max_epochs, use_early_stop, early_stop_max_epoch, verbose):
    # Print initial values
    batch = train_data.sample(batch_size) if batch_size else train_data
    train_error = model.eval_loss(batch)
    train_rmse = model.eval_rmse(batch)
    valid_rmse = model.eval_rmse(valid_data)
    print("[start] Train error: {:3f}, Train RMSE: {:3f}; Valid RMSE: {:3f}".format(train_error, train_rmse, valid_rmse))

    # Optimize
    prev_valid_rmse = float("Inf")
    early_stop_epochs = 0
    for epoch in range(max_epochs):
        # Run (S)GD
        shuffled_df = train_data.sample(frac=1)
        batches = chunk_df(shuffled_df, batch_size) if batch_size else [train_data]

        for batch_iter, batch in enumerate(batches):
            model.train_iteration(batch)

            # Evaluate
            train_error = model.eval_loss(batch)
            train_rmse = model.eval_rmse(batch)
            valid_rmse = model.eval_rmse(valid_data)
            if verbose:
                print("[{:d}-{:d}] Train error: {:3f}, Train RMSE: {:3f}; Valid RMSE: {:3f}".format(epoch, batch_iter, train_error, train_rmse, valid_rmse))

        # Checkpointing/early stopping
        if use_early_stop:
            early_stop_epochs += 1
            if valid_rmse < prev_valid_rmse:
                prev_valid_rmse = valid_rmse
                early_stop_epochs = 0
                saver.save(sess, model.model_filename)
            elif early_stop_epochs == early_stop_max_epoch:
                print("Early stopping ({} vs. {})...".format(prev_valid_rmse, valid_rmse))
                break
        else:
            saver.save(sess, model.model_filename)

def test(model, sess, saver, test_data, train_data=None, log=True):
    if train_data is not None:
        train_rmse = model.eval_rmse(train_data)
        if log:
            print("Final train RMSE: {}".format(train_rmse))

    test_rmse = model.eval_rmse(test_data)
    if log:
        print("Final test RMSE: {}".format(test_rmse))

    return test_rmse

def do_nnmf(
    mode, #'train' or 'test'
    param='{"lam":1.4841423900979607}', #lamda is regularize coefficient in model
    batch=25000,  #batch size as we know
    early_stop_max_epoch = 15, #the threshold number of non decresing loss
    max_epochs = 1000,   
    train_name='./data/mov_kaggle.train', 
    valid_name='./data/mov_kaggle.valid', 
    test_name= './data/mov_kaggle.test',
    verbose = True
    ):

    # Global args
    model_name = 'NNMF'
    mode = mode
    train_filename = train_name
    valid_filename = valid_name
    test_filename = test_name
    num_users = 10000
    num_items = 1000
    model_params = json.loads(param)
    delimiter = '\t'
    col_names = ['user_id', 'item_id', 'rating']
    batch_size = batch
    use_early_stop = True
    early_stop_max_epoch = early_stop_max_epoch
    max_epochs = max_epochs

    if mode in ('train', 'test'):
        with tf.Session() as sess:
            # Define computation graph & Initialize
            print('Building network & initializing variables')
            if model_name == 'NNMF':
                model = NNMF(num_users, num_items, **model_params)
            else:
                raise NotImplementedError("Model '{}' not implemented".format(model_name))

            model.init_sess(sess)
            saver = tf.train.Saver()

            # Train
            if mode in ('train', 'test'):
                # Process data
                print("Reading in data")
                train_data, valid_data, test_data = load_data(train_filename, valid_filename, test_filename,
                    delimiter=delimiter, col_names=col_names)

                if mode == 'train':
                    # Create model directory, if needed
                    if not os.path.exists(os.path.dirname(model.model_filename)):
                        os.makedirs(os.path.dirname(model.model_filename))

                    # Train
                    train(model, sess, saver, train_data, valid_data, batch_size=batch_size, max_epochs=max_epochs,
                          use_early_stop=use_early_stop, early_stop_max_epoch=early_stop_max_epoch, verbose= verbose)

                print('Loading best checkpointed model')
                # print(model.model_filename)
                saver.restore(sess, model.model_filename)

                test(model, sess, saver, test_data, train_data=train_data)
    else:
        raise Exception("Mode '{}' not available".format(mode))
