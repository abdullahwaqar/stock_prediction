# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from lib.data_fetcher import *
# import matplotlib.pyplot as plt

# data_frame = clean_data(get_stock_data())
# del data_frame[0]['Target']

# # print(data_frame[0])
# # print(clean_data(get_stock_data)[1])

# clf = LinearRegression()
# x_train, x_test, y_train, y_test = train_test_split(data_frame[0], data_frame[1], test_size = 0.2)

# # print(x_train)

# clf.fit(x_train, y_train)
# print(clf.score(x_test, y_test))

# y_predict = clf.predict(x_test)
# print(y_predict.shape)

# plt.style.use('ggplot')
# plt.plot(data_frame[2], data_frame[0]['Adj. Open'])
# plt.show()

from lib.Prediction_Models import PredictionModel
# from lib.data_fetcher import *

pm = PredictionModel()
pm.benchmark_model()

# df = get_historical_data('GOOGL','2005-01-01','2017-06-30')
# print(df)