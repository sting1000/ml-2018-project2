import numpy as np
from helpers import *
import scipy
import scipy.io
import scipy.sparse as sp
from utils import *


def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
        
    num_item, num_user = train.get_shape()

    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)

    # start by item features.
    item_nnz = train.getnnz(axis=1)
    item_sum = train.sum(axis=1)

    for ind in range(num_item):
        item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
    return user_features, item_features


def predict_mf(item_features, user_features, pred_input_path, pred_output_path):
    """Use item_features and user_features to make prediction according to pred_input_path"""
    mse = 0
    print("Predicting Now: ", pred_input_path)
    test = load_data(pred_input_path)
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    with open(pred_output_path, 'w') as output:
        output.write('Id,Prediction\n')
        for row, col in nnz_test:
            item_info = item_features[:, row]
            user_info = user_features[:, col]
            prediction = user_info.T.dot(item_info)
            prediction = min(5, prediction)
            prediction = max(1, prediction)
            output.write('r{}_c{},{}\n'.format(row + 1, col + 1, prediction))



def matrix_factorization_SGD(train, test,
    pred_input_path='./data/sample_submission.csv', # same format csv file as training data
    pred_output_path='./data/submit_sgd.csv',
    gamma = 0.012,
    num_features = 20,   # K in the lecture notes
    lambda_user = 0.01,
    lambda_item = 0.25,
    num_epochs = 50 ):    # number of full passes through the train set):
    """
    Matrix Factorization useing SGD algorithm. If test is None, function is in prediction mode
    :param train: training data, sparse matrix of shape (num_items, num_users)
    :param test: test data, sparse matrix of shape (num_items, num_users)
    :param pred_input_path: an input csv file path for prediction 
    :param pred_output_path:  an output csv file path to keep prediction result
    :param gamma: step size
    :param num_features:  number of latent features
    :param lambda_user:  regularization coefficient of user
    :param lambda_item: regularization coefficient of item
    :param num_epochs:  number of epochs for the optimization
    :return: Training RMSE and Test RMSE (if in prediciton mode, test rmse is nan)
    """

    errors = [0]
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            err = train[d, n] - user_info.T.dot(item_info)
    
            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)

        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        
        errors.append(rmse)

    with open("./data/train_sgd.csv", 'w') as output:
        output.write('Id,Prediction\n')
        for row, col in nz_train:
            item_info = item_features[:, row]
            user_info = user_features[:, col]
            prediction = user_info.T.dot(item_info)
            prediction = min(5, prediction)
            prediction = max(1, prediction)
            output.write('r{}_c{},{}\n'.format(row + 1, col + 1, prediction))
    print("---Training is done---")

    if test==None :
        print("Start prediction mode...")
        predict_mf(item_features, user_features, pred_input_path, pred_output_path)
        print("Predicting file is created.")
        return errors[-1], np.nan
    else:
        # evaluate the test error
        nz_row, nz_col = test.nonzero()
        nz_test = list(zip(nz_row, nz_col))
        rmse = compute_error(test, user_features, item_features, nz_test)
        print("test RMSE after running SGD:: {}.".format(rmse))
        return errors[-1], rmse

def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
    num_user = nnz_items_per_user.shape[0]
    num_feature = item_features.shape[0]
    lambda_I = lambda_user * sp.eye(num_feature)
    updated_user_features = np.zeros((num_feature, num_user))

    for user, items in nz_user_itemindices:
        # extract the columns corresponding to the prediction for given item
        M = item_features[:, items]
        
        # update column row of user features
        V = M @ train[items, user]
        A = M @ M.T + nnz_items_per_user[user] * lambda_I
        X = np.linalg.solve(A, V)
        updated_user_features[:, user] = np.copy(X.T)
    return updated_user_features

def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    num_item = nnz_users_per_item.shape[0]
    num_feature = user_features.shape[0]
    lambda_I = lambda_item * sp.eye(num_feature)
    updated_item_features = np.zeros((num_feature, num_item))

    for item, users in nz_item_userindices:
        # extract the columns corresponding to the prediction for given user
        M = user_features[:, users]
        V = M @ train[item, users].T
        A = M @ M.T + nnz_users_per_item[item] * lambda_I
        X = np.linalg.solve(A, V)
        updated_item_features[:, item] = np.copy(X.T)
    return updated_item_features


def matrix_factorization_ALS(train, test, 
    pred_input_path='./data/sample_submission.csv',
    pred_output_path='./data/submit_als.csv',
    num_features = 20,   
    lambda_user = 0.014,   
    lambda_item = 0.575,   
    stop_criterion = 1e-5):

    """
    Alternating Least Squares (ALS) algorithm. If test is None, function is in prediction mode
    :param train: training data, sparse matrix of shape (num_items, num_users)
    :param test: test data, sparse matrix of shape (num_items, num_users)
    :param pred_input_path: an input csv file path for prediction 
    :param pred_output_path:  an output csv file path to keep prediction result
    :param num_features:  number of latent features
    :param lambda_user:  regularization coefficient of user
    :param lambda_item: regularization coefficient of item
    :param stop_criterion:  a boolean value, True means function is for prediction
    :return: Training RMSE and Test RMSE (if in prediciton mode, test rmse is nan)
    """
    change = 1
    error_list = [0, 0]
    
    # set seed
    np.random.seed(988)
    # init ALS
    user_features, item_features = init_MF(train, num_features)
    
    # get the number of non-zero ratings for each user and item
    nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=0), train.getnnz(axis=1)
    
    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)

    # run ALS
    print("Start the ALS algorithm...")
    while change > stop_criterion:
        # update user feature & item feature
        user_features = update_user_feature(
            train, item_features, lambda_user,
            nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(
            train, user_features, lambda_item,
            nnz_users_per_item, nz_item_userindices)

        error = compute_error(train, user_features, item_features, nz_train)
        print("RMSE on training set: {}.".format(error))
        error_list.append(error)
        change = np.fabs(error_list[-1] - error_list[-2])

    #predic training set 
    nnz_row, nnz_col = train.nonzero()
    nnz_train = list(zip(nnz_row, nnz_col))
    with open("./data/train_als.csv", 'w') as output:
        output.write('Id,Prediction\n')
        for row, col in nnz_train:
            item_info = item_features[:, row]
            user_info = user_features[:, col]
            prediction = user_info.T.dot(item_info)
            prediction = min(5, prediction)
            prediction = max(1, prediction)
            output.write('r{}_c{},{}\n'.format(row + 1, col + 1, prediction))
    print("---Training is done---")

    if test==None :
        print("Start prediction mode...")
        predict_mf(item_features, user_features, pred_input_path, pred_output_path)
        print("Predicting file is created.")
        return error, np.nan
    else:
        # evaluate the test error
        nnz_row, nnz_col = test.nonzero()
        nnz_test = list(zip(nnz_row, nnz_col))
        rmse = compute_error(test, user_features, item_features, nnz_test)
        print("test RMSE after running ALS: {v}.".format(v=rmse))
        return error, rmse



