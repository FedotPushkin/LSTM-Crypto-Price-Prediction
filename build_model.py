from tensorflow import keras
import keras.regularizers as reg
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, AlphaDropout
def build_model(x_adj, learning_rate):
    # first layer
    model = Sequential()
    model.add(LSTM(100,
                   input_shape=(x_adj.shape[1], x_adj.shape[2]),
                   return_sequences=True,
                   kernel_regularizer=reg.L1(0.005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))

    # second layer
    model.add(LSTM(100, return_sequences=True, kernel_regularizer=reg.L1(0.005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False, kernel_regularizer=reg.L1(0.005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))

    model.add(Dense(20, activation='relu', kernel_regularizer=reg.L1(0.005)))
    #model.add(AlphaDropout(0.5))
    #model.add(Dropout(0.1))
    # fourth layer and output
    #model.add(Dense(1, activation='linear'))    # kernel_regularizer=regularizers.L1(0.01)

    optimizer = keras.optimizers.Nadam(learning_rate=learning_rate,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       decay=0.004)

    #model.compile(loss='mse', metrics=[keras.metrics.RootMeanSquaredError()], optimizer = 'rmsprop' )

    # model.add(Dense(20, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # compile layers
    model.compile(loss='categorical_crossentropy',
                 optimizer='rmsprop',
                 metrics=['accuracy'])

    return model