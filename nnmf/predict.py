#!/usr/bin/env python2.7
from __future__ import absolute_import, print_function
"""Generates predictions."""
# Standard modules
import argparse, json
# Third party modules
import tensorflow as tf
# Package modules
from nnmf.models import NNMF

import pandas as pd
import numpy as np

def predict_nnmf(filename, model_params = '{"lam":1.4841423900979607}'):


    model_name = 'NNMF'
    model_params = json.loads(model_params)
    num_users = 10000
    num_items = 1000
    # user_id = args.user
    # item_id = args.item

    predict_filename = filename #'../data/kaggle/split/mov_kaggle.train'
    predict_data = pd.read_csv(predict_filename, delimiter='\t', header=None, names=['user_id', 'item_id', 'rating'])
    predict_data['user_id'] = predict_data['user_id'] - 1
    predict_data['item_id'] = predict_data['item_id'] - 1

    submit_data  = pd.read_csv('./data/sample_submission.csv')
    # submit_data  = pd.DataFrame()

    print('Building network & initializing variables')
    if model_name == 'NNMF':
        model = NNMF(num_users, num_items, **model_params)
    else:
        raise NotImplementedError("Model '{}' not implemented".format(model_name))

    with tf.Session() as sess:
        model.init_sess(sess)
        saver = tf.train.Saver()

        print('Loading model: ', model.model_filename)
        saver.restore(sess, model.model_filename)
        # user_id, item_id = args.user, args.item
        submit_data['Prediction'] = model.np.round(predict(predict_data['user_id'], predict_data['item_id']))#
        submit_data.loc[submit_data['Prediction']>5, 'Prediction']= 5
        submit_data.loc[submit_data['Prediction']<1, 'Prediction']= 1
        submit_data.to_csv('submit.csv', index=False)
    #print("Predicted rating for user '{}' & item '{}': {}".format(user_id, item_id, rating))
