import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error
import keras
from keras import losses

def preprocess_for_neural_net(dataset):
    dataset['user'] = dataset['Id'].apply( lambda x : x.split('_')[0][1:]).astype('int')
    dataset['movie'] = dataset['Id'].apply(lambda x : x.split('_')[1][1:]).astype('int')
    dataset.drop('Id',axis=1,inplace = True)
    dataset['rating'] = dataset['Prediction']
    dataset.drop('Prediction', axis=1, inplace=True)
    return dataset


def neural_network(n_users, n_movies, n_latent_factors_user, n_latent_factors_movie):

	movie_input = keras.layers.Input(shape=[1],name='Item')
	movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors_movie, name='Movie-Embedding')(movie_input)
	movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
	movie_vec = keras.layers.Dropout(0.2)(movie_vec)


	user_input = keras.layers.Input(shape=[1],name='User')
	user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, n_latent_factors_user,name='User-Embedding')(user_input))
	user_vec = keras.layers.Dropout(0.2)(user_vec)


	concat = keras.layers.concatenate([movie_vec, user_vec],axis=1)
	concat_dropout = keras.layers.Dropout(0.2)(concat)
	dense = keras.layers.Dense(100,activation = 'relu', name='FullyConnected')(concat)
	dropout_1 = keras.layers.Dropout(0.15,name='Dropout')(dense)
	dense_2 = keras.layers.Dense(70,activation = 'relu',name='FullyConnected-1')(concat)
	dropout_2 = keras.layers.Dropout(0.2,name='Dropout')(dense_2)
	dense_3 = keras.layers.Dense(40,activation = 'relu',name='FullyConnected-2')(dense_2)
	dropout_3 = keras.layers.Dropout(0.1,name='Dropout')(dense_3)
	dense_4 = keras.layers.Dense(10,name='FullyConnected-3', activation='relu')(dense_3)
	dropout_4 = keras.layers.Dropout(0.25,name='Dropout')(dense_4)

	result = keras.layers.Dense(1, activation='relu',name='Activation')(dropout_4)
	model = keras.Model([user_input, movie_input], result)

	return model

