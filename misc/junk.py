def linear_regression_(self):
        del self.data_frame[0]['Target']
        clf = LinearRegression()
        x_train, x_test, y_train, y_test = train_test_split(self.data_frame[0], self.data_frame[1], test_size = 0.2)
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        plt.style.use('ggplot')
        plt.plot(self.data_frame[2], self.data_frame[0]['Adj. Open'])
        plt.show()
        # self.data_frame[0].columns = ['0','1', '2', '3', '4', '5']
        # self.data_frame[1] = self.data_frame[0]

        # print(self.data_frame[1].head())

    def linear_regression(self):
        new_target = ((self.data_frame[0]['Adj. Open'] - self.data_frame[1]) / self.data_frame[0]['Adj. Open']) * 100
        self.data_frame[1] = new_target
        self.data_frame[0] = scale(self.data_frame[0])
        self.data_frame[0] = pd.DataFrame(self.data_frame[0])

        self.data_frame[1] = self.data_frame[0][5]
        del self.data_frame[0][5]

        x_train, x_test, y_train, y_test = train_test_split(self.data_frame[0], self.data_frame[1], test_size = 0.2)
        clf = LinearRegression()
        clf.fit(x_train, y_train)
        clf.score(x_test, y_test)
        print(clf.coef_)

    def LSTM(self):
        data = pd.read_csv('data/HistoricalQuotes.csv')
        #* Scale Data
        scl = MinMaxScaler()
        data = np.array(data)
        print(data.shape[0], 1)
        data = data.flatten().reshape(data.shape[0], 1)
        data = scl.fit_transform(data)
        print(data)
        # x_train, x_test, y_train, y_test = train_test_split(self.data_frame[0], self.data_frame[1], test_size = 0.2, random_state = 4)
        # # print(x_test)
        # #* RNN Model Setup
        # print(x_train.shape)
        # model = Sequential()
        # model.add(LSTM(64, input_dim=2 , input_shape=x_train.shape))
        # model.add(Dense(1))
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.summary()
        # history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), shuffle=False)
        # print(history)