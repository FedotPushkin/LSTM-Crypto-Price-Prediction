from tensorflow import keras
import keras_tuner
import keras.regularizers as reg
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, AlphaDropout, Attention
def build_model(x_adj, learning_rate):
    # first layer
    model = Sequential()
    model.add(LSTM(20,
                   input_shape=(x_adj.shape[1], x_adj.shape[2]),
                   return_sequences=True,
                   recurrent_regularizer=reg.L1(0.000001),
                   kernel_regularizer=reg.L1(0.000001)))#,
                   #recurrent_dropout=0.05))
    model.add(BatchNormalization())
    #model.add(Dropout(0.15))

    # second layer
    #model.add(LSTM(100, return_sequences=True, kernel_regularizer=reg.L1(0.005)))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(LSTM(40, return_sequences=False, recurrent_regularizer=reg.L1(0.000001), kernel_regularizer=reg.L1(0.000001), ))
    model.add(BatchNormalization())
    #model.add(Dropout(0.15))

    model.add(Dense(20, activation='relu', kernel_regularizer=reg.L1(0.0001)))
    #model.add(AlphaDropout(0.5))
    model.add(Dropout(0.001))
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
    model.compile(loss='binary_crossentropy',
                 optimizer='rmsprop',
                 metrics=['accuracy'])

    return model


def build_model_bayesian(hp):
    model = Sequential()
    model.add(keras.layers.LSTM(units=hp.Int('units', min_value=40, max_value=800, step=20),
                                dropout=hp.Float('droput', min_value=0.15, max_value=0.99, step=0.05),
                                recurrent_dropout=hp.Float('redroput', min_value=0.05, max_value=0.99, step=0.05),
                                activation='relu',
                                return_sequences=True,
                                input_shape=(30, 1)))
    Attention()
    model.add(keras.layers.LSTM(units=hp.Int('units', min_value=40, max_value=800, step=20),
                                dropout=hp.Float('droput', min_value=0.15, max_value=0.99, step=0.05),
                                activation='relu', return_sequences=True))
    Attention()
    model.add(keras.layers.LSTM(units=hp.Int('units', min_value=40, max_value=800, step=20), activation='relu'))
    model.add(keras.layers.Dense(1))

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(
        hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-7, 1e-10])))
    return model


bayesian_opt_tuner = keras_tuner.BayesianOptimization(
    build_model_bayesian,
    objective='val_loss',
    max_trials=200,
    executions_per_trial=1,
    project_name='timeseries_bayes_opt_POC',
    overwrite=True, )

def opt_search(x_adj, y_adj):

    bayesian_opt_tuner.search(x= x_adj, y=y_adj,
                          epochs=300,
                          # validation_data=(xval ,xval),
                          validation_split=0.95,
                          validation_steps=30,
                          steps_per_epoch=30,
                          callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                      patience=4,
                                                                      verbose=1,
                                                                      restore_best_weights=True),
                                     keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                          factor=0.1,
                                                                          patience=2,
                                                                          verbose=1,
                                                                          min_delta=1e-5,
                                                                          mode='min')]
                          )