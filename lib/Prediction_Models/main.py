import math
import time
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

from ..Linear_Regression import *
from ..LTSM_Model import *
from ..preprocess_data import remove_data, get_normalised_data
from ..stock_data_parser import train_test_split_linear_regression, train_test_split_lstm
from ..visualize import *

class PredictionModel:

    def __init__(self):
        self.data = pd.read_csv('data/google.csv')

    def get_stock(self):
        """ Remove Unncessary data, i.e., Date and High value from stock data """
        stocks = remove_data(self.data)
        return stocks

    def normalise_stock_data(self):
        """ Normalising the data using minmaxscaler function implemented in lib/preprocess_data """
        stocks = self.get_stock()
        return get_normalised_data(stocks)
        # plot_basic(stocks)

    """
    @desc: Benchmark model implementation
           A simple linear regression model is implemented to check the accuracy.
    @return: Ploted predicted values vs real values with accuracy score
    """
    def linear_regression_model(self):
        # print(self.normalise_stock_data())
        """ Split data into train and test pair """
        X_train, X_test, y_train, y_test, label_range= train_test_split_linear_regression(self.normalise_stock_data())

        """ Train a Linear regressor model on training set and get prediction """
        model = build_model(X_train, y_train)

        """ Get prediction on test set """
        predictions = predict_prices(model, X_test, label_range)

        """ Plot the predicted values against actual """
        plot_prediction(y_test, predictions)

        """ Measure accuracy of the prediction """
        trainScore = mean_squared_error(X_train, y_train)
        print('Train Score: %.4f MSE (%.4f RMSE)' % (trainScore, math.sqrt(trainScore)))

        testScore = mean_squared_error(predictions, y_test)
        print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))

    def basic_LTSM_model(self):
        stocks_data = pd.read_csv('data/google_preprocessed.csv')
        stocks_data = stocks_data.drop(['Item'], axis=1)

        """ Split data into train and test pair """
        X_train, X_test, y_train, y_test = train_test_split_lstm(stocks_data, 5)
        unroll_length = 50
        X_train = unroll(X_train, unroll_length)
        X_test = unroll(X_test, unroll_length)
        y_train = y_train[-X_train.shape[0]:]
        y_test = y_test[-X_test.shape[0]:]

        """ Build a basic Long-Short Term Memory model """
        model = build_basic_model(input_dim = X_train.shape[-1],output_dim = unroll_length, return_sequences=True)

        """ Compile the model """
        start = time.time()
        model.compile(loss='mean_squared_error', optimizer='adam')
        print('compilation time : ', time.time() - start)

        """ Train the model """
        model.fit(
            X_train,
            y_train,
            epochs=1,
            validation_split=0.05
        )

        """ Make prediction using test data """
        predictions = model.predict(X_test)

        """ Plot the results """
        plot_prediction(y_test, predictions)

        """ Get the test score. """
        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))

        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))

    def improved_LTSM_model(self):
        stocks_data = pd.read_csv('data/google_preprocessed.csv')
        stocks_data = stocks_data.drop(['Item'], axis=1)
        unroll_length = 50

        """ Split data into train and test pair """
        X_train, X_test, y_train, y_test = train_test_split_lstm(stocks_data, 5)
        X_train = unroll(X_train, unroll_length)
        X_test = unroll(X_test, unroll_length)
        y_train = y_train[-X_train.shape[0]:]
        y_test = y_test[-X_test.shape[0]:]
        # Set up hyperparameters
        batch_size = 100
        epochs = 5

        """ Build an improved LSTM model """
        model = build_improved_model(X_train.shape[-1], output_dim = unroll_length, return_sequences=True)

        start = time.time()
        #final_model.compile(loss='mean_squared_error', optimizer='adam')
        model.compile(loss='mean_squared_error', optimizer='adam')
        print('compilation time : ', time.time() - start)

        """ Train improved LSTM model """
        model.fit(X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_split=0.05
         )

        """ Make prediction on improved LSTM model """
        predictions = model.predict(X_test, batch_size=batch_size)

        """ Plot the results """
        plot_lstm_prediction(y_test,predictions)

        """ Get the test score """
        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))

        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))