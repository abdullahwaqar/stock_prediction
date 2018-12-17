import math
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
from ..stock_data_parser import train_test_split_linear_regression
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
