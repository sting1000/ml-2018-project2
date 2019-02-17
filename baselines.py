import numpy as np
from helpers import *
import scipy
import scipy.io
import scipy.sparse as sp


def baseline_global_mean(train, test, 
    pred_input_path='./data/sample_submission.csv',
    pred_output_path='./data/submit_base_glob.csv' ):
    """
    baseline method: use the global mean. If test is None, the function is in prediction mode
    :param train: training data, sparse matrix of shape (num_items, num_users)
    :param test: test data, sparse matrix of shape (num_items, num_users)
    :param pred_input_path: an input csv file path for prediction 
    :param pred_output_path:  an output csv file path to keep prediction result
    :return: RMSE
    """
    # find the non zero ratings in the train
    nonzero_train = train[train.nonzero()]

    # calculate the global mean
    global_mean_train = nonzero_train.mean()

    if test == None:
        print("Start prediction mode...")
        global_mean_train = min(5, global_mean_train)
        global_mean_train = max(1, global_mean_train)
        test = load_data(pred_input_path)
        nnz_row, nnz_col = test.nonzero()
        nnz_test = list(zip(nnz_row, nnz_col))
        with open(pred_output_path, 'w') as output:
            output.write('Id,Prediction\n')
            for row, col in nnz_test:
                output.write('r{}_c{},{}\n'.format(row + 1, col + 1, global_mean_train))
        print("Done!")
    else:
        # find the non zero ratings in the test
        nonzero_test = test[test.nonzero()].todense()

        # predict the ratings as global mean
        mse = calculate_mse(nonzero_test, global_mean_train)
        rmse = np.sqrt(1.0 * mse / nonzero_test.shape[1])
        print("test RMSE of baseline using the global mean: {v}.".format(v=rmse))
        return rmse

def baseline_user_mean(train, test,
    pred_input_path='./data/sample_submission.csv',
    pred_output_path='./data/submit_base_user.csv' ):
    """
    baseline method: use the user means as the prediction.If test is None, the function is in prediction mode
    :param train: training data, sparse matrix of shape (num_items, num_users)
    :param test: test data, sparse matrix of shape (num_items, num_users)
    :param pred_input_path: an input csv file path for prediction 
    :param pred_output_path:  an output csv file path to keep prediction result
    :return: RMSE if test is not None else nothing
    """
    mse = 0
    num_items, num_users = train.shape
    means = [0 for _ in range(train.shape[1])]

    for user_index in range(num_users):
        # find the non-zero ratings for each user in the training dataset
        train_ratings = train[:, user_index]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]
        
        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            user_train_mean = nonzeros_train_ratings.mean() 
        else:
            continue
        if test == None:
            means[user_index] =  user_train_mean

        else:
            # find the non-zero ratings for each user in the test dataset
            test_ratings = test[:, user_index]
            nonzeros_test_ratings = test_ratings[test_ratings.nonzero()].todense()
            
            # calculate the test error 
            mse += calculate_mse(nonzeros_test_ratings, user_train_mean)
    if test == None:
        print("Start prediction mode...")
        test = load_data(pred_input_path)
        nnz_row, nnz_col = test.nonzero()
        nnz_test = list(zip(nnz_row, nnz_col))
        with open(pred_output_path, 'w') as output:
            output.write('Id,Prediction\n')
            for row, col in nnz_test:
                output.write('r{}_c{},{}\n'.format(row + 1, col + 1, means[col]))
        print("Done!")
    else:
        rmse = np.sqrt(1.0 * mse / test.nnz)
        print("test RMSE of the baseline using the user mean: {v}.".format(v=rmse))
        return rmse

def baseline_item_mean(train, test,
    pred_input_path='./data/sample_submission.csv',
    pred_output_path='./data/submit_base_item.csv' ):
    """
    baseline method: use the item means as the prediction.If test is None, the function is in prediction mode
    :param train: training data, sparse matrix of shape (num_items, num_users)
    :param test: test data, sparse matrix of shape (num_items, num_users)
    :param pred_input_path: an input csv file path for prediction 
    :param pred_output_path:  an output csv file path to keep prediction result
    :return: RMSE if test is not None else nothing
    """
    mse = 0
    num_items, num_users = train.shape
    means = [0 for _ in range(train.shape[0])]
    
    for item_index in range(num_items):
        # find the non-zero ratings for each item in the training dataset
        train_ratings = train[item_index, :]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            item_train_mean = nonzeros_train_ratings.mean()
        else:
            continue

        if test == None:
            means[item_index] =  item_train_mean
        else:
            # find the non-zero ratings for each movie in the test dataset
            test_ratings = test[item_index, :]
            nonzeros_test_ratings = test_ratings[test_ratings.nonzero()].todense()
            
            # calculate the test error 
            mse += calculate_mse(nonzeros_test_ratings, item_train_mean)
    if test == None:
        print("Start prediction mode...")
        test = load_data(pred_input_path)
        nnz_row, nnz_col = test.nonzero()
        nnz_test = list(zip(nnz_row, nnz_col))
        with open(pred_output_path, 'w') as output:
            output.write('Id,Prediction\n')
            for row, col in nnz_test:
                output.write('r{}_c{},{}\n'.format(row + 1, col + 1, means[row]))
        print("Done!")
    else:       
        rmse = np.sqrt(1.0 * mse / test.nnz)
        print("test RMSE of the baseline using the item mean: {v}.".format(v=rmse))
        return rmse