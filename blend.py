
import numpy as np
from helpers import *
import scipy
import scipy.io
import scipy.sparse as sp

def blend(m_train, y_train=None, alpha=0.1, bool_pred=False, w=None):
    """
    blend uses Ridge Regression to find weights for each algorithm. If bool_pred is True, this funciotn require 'w ' and will predict using m_train
    :param m_train: is an array with shape N*(model_num)
    :param y_train: is an N*1 array of true values of m_train. 
    :param alpha: the hyperparameter of ridge regression
    :param bool_pred:  a boolean value, True means function is for prediction
    :param w: the blending weights after training (the result of training model)
    :return: w and RMSE for training mode, N*1 prediction array for predictoin mode
    """
    if  bool_pred:
        print("Start blend-Predicting mode(return prediction array)...")
        y_predict_train = m_train.dot(w)
        # Cut predictions that are too high and too low
        for i in range(len(y_predict_train)):
            y_predict_train[i] = min(5, np.round(y_predict_train[i]))
            y_predict_train[i] = max(1, np.round(y_predict_train[i]))
        print("Done!")
        return y_predict_train

    else:
        print("Start blend-Training mode(return w and RMSE)...")
        w = np.linalg.solve(m_train.T.dot(m_train) + alpha * np.eye(m_train.shape[1]), m_train.T.dot(y_train))
        y_predict_train = m_train.dot(w)
        # Cut predictions that are too high and too low
        for i in range(len(y_predict_train)):
            y_predict_train[i] = min(5, np.round(y_predict_train[i]))
            y_predict_train[i] = max(1, np.round(y_predict_train[i]))
        RMSE = np.sqrt(np.mean((y_train - y_predict_train) ** 2))
        print("Done!")
        return w, RMSE