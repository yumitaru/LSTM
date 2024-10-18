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

# def df_to_X_y(df, window_size=5):
#     df_np = df.to_numpy()
#     X = []
#     y = []
#     for i in range(len(df_np)-window_size):
#         row = [[a] for a in df_np[i:i+window_size]]
#         X.append(row)
#         label = df_np[i+5]
#         y.append(label)
#     return np.array(X), np.array(y)

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
                                                                    # 'Battery cell voltage',
                                                                        'Going to ID',
                                                                        'Y-coordinate',
                                                                        'X-coordinate'
                                                                        ],
                                                                    engine='python')

    dataframe['X-coordinate'] = pd.to_numeric(dataframe['X-coordinate'], errors='coerce')
    # x_coor = pd.to_numeric(dataframe['Y-coordinate'], errors='coerce')
    # y_coor = pd.to_numeric(dataframe['Y-coordinate'], errors='coerce')

    look_back = 1
    dataframe = dataframe.values
    dataframe = dataframe.astype('float32')

    dataframe = dataframe[20000:22500:5]
    # print(dataframe.shape)


    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataframe)

    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # print(len(train), len(test))

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # print(trainX.shape)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1]))
    # print(trainX.shape)
    # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # print(trainX.shape[0])
    # print(trainX.shape[1])
    # print(trainX.shape)
    # print(dataset.shape)
    # print(dataframe[0,1])
    # print(dataset.shape)
    # print(trainX[0])

    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(3))
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
    # trainPredict = scaler.inverse_transform(np.concatenate((trainPredict, np.zeros((trainPredict.shape[0], 1))), axis=1))
    # trainY = scaler.inverse_transform(np.concatenate((trainY.reshape(-1, 1), np.zeros((trainY.shape[0], 1))), axis=1))
    # testPredict = scaler.inverse_transform(np.concatenate((testPredict, np.zeros((testPredict.shape[0], 1))), axis=1))
    # testY = scaler.inverse_transform(np.concatenate((testY.reshape(-1, 1), np.zeros((testY.shape[0], 1))), axis=1))
    # calculate root mean squared error
    # trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    # print('Train Score: %.2f RMSE' % (trainScore))
    # testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    # print('Test Score: %.2f RMSE' % (testScore))
    # print(trainPredict.shape)
    # print(trainY.shape)

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

    # print(trainPredict[5][0])
    # print(trainPredict[5][1])
    # print(trainPredict[6][1])
    # print(trainPredict[7][1])
    # print(trainPredict.shape)
    # print(dataframe.shape)



    # dataframe['X-coordinate'] = pd.to_numeric(dataframe['X-coordinate'], errors='coerce')
    # print(dataframe.to_string())
    # plt.plot(dataframe['Battery cell voltage'])
    # plt.show()
