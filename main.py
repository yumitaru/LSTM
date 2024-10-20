import pickle
import pandas
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import LSTM


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        tempX = []
        tempY = []
        for j in range(len(dataset[i])):
            a = dataset[i:(i+look_back), j]
            tempX.append(a)
            tempY.append(dataset[i + look_back, j])
        dataX.append(tempX)
        dataY.append(tempY)
    return np.array(dataX), np.array(dataY)



if __name__ == '__main__':
    tf.random.set_seed(7)
    dataframe = pd.read_csv(r'6000_frames_20221124+25_new.pkl', usecols=[
                                                                    # 'id_val',
                                                                    # 'Timestamp',
                                                                    'Battery cell voltage',
                                                                        'Heading',
                                                                        'Going to ID',
                                                                        'Y-coordinate',
                                                                        'X-coordinate'
                                                                        ],
                                                                    engine='python')

    dataframe['X-coordinate'] = pd.to_numeric(dataframe['X-coordinate'], errors='coerce')


    look_back = 3
    dataframe = dataframe.values
    dataframe = dataframe.astype('float32')

    dataframe = dataframe[20000:22500:5]


    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataframe)

    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)


    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(5))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # print(trainY.shape)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    # print(trainPredict.shape)
    # print(dataframe.shape)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)

    # calculate root mean squared error
    # trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    # print('Train Score: %.2f RMSE' % (trainScore))
    # testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    # print('Test Score: %.2f RMSE' % (testScore))


    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

